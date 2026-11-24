"""
Unit tests for Week 2: Advanced Sequential Modeling with Temporal Attention (CO3).
Validates:
1. TemporalAttention mathematical constraints (Softmax normalization, context vector dimensions).
2. AttentionGRU & AttentionLSTM dual-head outputs (Risk in [0, 1], Stage logits, Attention weights).
3. Joint multi-task gradient backpropagation through both heads.
4. Sequence generator multi-class stage label generation.
5. Explainability reporting and ASCII visualization utilities.
"""

import os
import sys
import torch
import pytest
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from src.features.sequence_generator import prepare_sequential_data, STAGE_NAMES, STAGE_MAPPING
from src.models.attention_model import TemporalAttention, AttentionGRU, AttentionLSTM
from src.models.attention_visualizer import make_ascii_bar, format_attention_report, generate_prediction_narrative

CSV_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "pipeline_telemetry.csv")

def test_temporal_attention_mathematical_properties():
    batch_size = 12
    seq_len = 4
    hidden_dim = 16
    
    layer = TemporalAttention(hidden_dim=hidden_dim, attention_dim=8)
    dummy_input = torch.randn(batch_size, seq_len, hidden_dim, requires_grad=True)
    
    context, weights = layer(dummy_input)
    
    # 1. Dimension checks
    assert context.shape == (batch_size, hidden_dim), "Context vector shape mismatch"
    assert weights.shape == (batch_size, seq_len), "Attention weights shape mismatch"
    
    # 2. Probability distribution constraint: weights must sum to 1.0 per sequence
    weight_sums = weights.sum(dim=1)
    expected_sums = torch.ones(batch_size)
    assert torch.allclose(weight_sums, expected_sums, atol=1e-5), "Attention weights must sum to 1.0"
    
    # 3. Non-negativity constraint
    assert (weights >= 0.0).all(), "Attention weights must be non-negative"
    
    # 4. Gradient flow
    loss = context.sum()
    loss.backward()
    assert dummy_input.grad is not None, "Gradients failed to backpropagate through attention layer"

def test_attention_gru_dual_head_forward():
    batch_size = 8
    seq_len = 3
    input_dim = 4
    num_classes = 5
    
    model = AttentionGRU(input_dim=input_dim, hidden_dim=20, attention_dim=12, num_classes=num_classes)
    dummy_x = torch.randn(batch_size, seq_len, input_dim)
    
    risk_prob, cause_logits, attn_weights = model(dummy_x)
    
    # Check Head 1 (Binary Risk)
    assert risk_prob.shape == (batch_size,), "Risk probability shape mismatch"
    assert (risk_prob >= 0.0).all() and (risk_prob <= 1.0).all(), "Risk probabilities out of [0, 1] bounds"
    
    # Check Head 2 (Root Cause Stage Logits)
    assert cause_logits.shape == (batch_size, num_classes), "Cause logits shape mismatch"
    
    # Check Attention Weights
    assert attn_weights.shape == (batch_size, seq_len), "Attention weights shape mismatch"
    assert torch.allclose(attn_weights.sum(dim=1), torch.ones(batch_size), atol=1e-5)

def test_attention_lstm_forward():
    batch_size = 6
    seq_len = 5
    input_dim = 4
    num_classes = 5
    
    model = AttentionLSTM(input_dim=input_dim, hidden_dim=16, num_classes=num_classes)
    dummy_x = torch.randn(batch_size, seq_len, input_dim)
    
    risk_prob, cause_logits, attn_weights = model(dummy_x)
    assert risk_prob.shape == (batch_size,)
    assert cause_logits.shape == (batch_size, num_classes)
    assert attn_weights.shape == (batch_size, seq_len)

def test_multitask_gradient_backprop():
    batch_size = 4
    seq_len = 3
    input_dim = 4
    
    model = AttentionGRU(input_dim=input_dim, hidden_dim=16, num_classes=5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    dummy_x = torch.randn(batch_size, seq_len, input_dim)
    target_risk = torch.tensor([0.0, 1.0, 0.0, 1.0])
    target_stage = torch.tensor([0, 1, 0, 3], dtype=torch.long)
    
    risk_prob, cause_logits, _ = model(dummy_x)
    loss = torch.nn.BCELoss()(risk_prob, target_risk) + torch.nn.CrossEntropyLoss()(cause_logits, target_stage)
    
    optimizer.zero_grad()
    loss.backward()
    
    # Ensure gradients computed for all components
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Gradient missing for parameter {name}"
        assert not torch.isnan(param.grad).any(), f"NaN gradient in parameter {name}"

def test_sequence_generator_stage_labels():
    data = prepare_sequential_data(CSV_PATH, window_size=3)
    assert "y_stage_train" in data
    assert "y_stage_val" in data
    assert "y_stage_test" in data
    
    # Ensure classes are within 0 to 4
    for key in ["y_stage_train", "y_stage_val", "y_stage_test"]:
        stages = data[key].numpy()
        assert (stages >= 0).all() and (stages < len(STAGE_NAMES)).all(), f"Invalid stage label in {key}"

def test_attention_visualization_utilities():
    weights = np.array([0.15, 0.35, 0.50])
    report = format_attention_report(weights)
    assert "Temporal Attention Distribution" in report
    assert "Run t-1" in report
    assert "50.0%" in report
    
    bar = make_ascii_bar(0.5, max_val=1.0, length=10)
    assert len(bar) == 10
    
    narrative = generate_prediction_narrative(
        risk_prob=0.85,
        cause_probs=np.array([0.05, 0.80, 0.05, 0.05, 0.05]),
        attn_weights=weights,
        stage_names=STAGE_NAMES
    )
    assert "CRITICAL HIGH" in narrative
    assert "API_INGESTION" in narrative

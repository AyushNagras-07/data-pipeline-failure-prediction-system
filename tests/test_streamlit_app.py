"""
Unit tests for Week 4: Streamlit Web Dashboard Application (CO5).
Validates:
1. Application integrity and required script existence.
2. Model loading and pipeline asset verification.
3. Real-time inference simulation pipeline.
4. Report figures availability for UI rendering.
"""

import os
import sys
import torch
import pytest
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from src.models.attention_model import AttentionGRU
from src.features.sequence_generator import FEATURE_COLUMNS, STAGE_NAMES

def test_app_and_runner_scripts_exist():
    app_py = os.path.join(PROJECT_ROOT, "app.py")
    run_week4_py = os.path.join(PROJECT_ROOT, "run_week4.py")
    assert os.path.exists(app_py), "app.py missing"
    assert os.path.exists(run_week4_py), "run_week4.py missing"

def test_app_model_and_scaler_loading():
    ckpt_path = os.path.join(PROJECT_ROOT, "models", "best_attention_model.pth")
    assert os.path.exists(ckpt_path), f"Checkpoint not found at {ckpt_path}"
    
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert "model_state_dict" in ckpt
    assert "scaler" in ckpt
    
    scaler = ckpt["scaler"]
    model = AttentionGRU(input_dim=4, hidden_dim=24, attention_dim=16, num_classes=5)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    # Test sample inference
    dummy_telemetry = np.array([
        [2.5, 10, 0, 85.0],
        [4.2, 10, 1, 98.0],
        [8.5, 10, 2, 125.0]
    ])
    scaled = scaler.transform(dummy_telemetry)
    tensor = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        risk_prob, cause_logits, attn_weights = model(tensor)
        
    assert 0.0 <= float(risk_prob.item()) <= 1.0
    assert cause_logits.shape == (1, 5)
    assert attn_weights.shape == (1, 3)
    assert np.isclose(float(attn_weights.sum().item()), 1.0, atol=1e-5)

def test_figures_exist_for_ui():
    fig_dir = os.path.join(PROJECT_ROOT, "reports", "figures")
    expected_figures = [
        "1_architecture_comparison_metrics.png",
        "2_window_size_ablation.png",
        "3_training_convergence_curves.png",
        "4_roc_curves.png",
        "5_confusion_matrix_subsystems.png"
    ]
    for fig in expected_figures:
        p = os.path.join(fig_dir, fig)
        assert os.path.exists(p), f"Figure {fig} missing in {fig_dir}"
        assert os.path.getsize(p) > 1000, f"Figure {fig} is empty"

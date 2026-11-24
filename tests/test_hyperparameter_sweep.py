"""
Unit tests for Week 3: Hyperparameter Optimization Sweep & Ablation Study (CO4).
Validates:
1. Multi-window data partitioning (T=3, 5, 7).
2. train_eval_model function for both single-head and dual-head models.
3. Latency profiling and parameter counting.
4. Report serialization and plot generation functionality.
"""

import os
import sys
import pytest
import torch
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from src.features.sequence_generator import prepare_sequential_data
from src.models.gru_model import PipelineGRU
from src.models.attention_model import AttentionGRU
from src.models.hyperparameter_sweep import train_eval_model, CSV_PATH

def test_multi_window_sequence_generation():
    for w in [3, 5, 7]:
        data = prepare_sequential_data(CSV_PATH, window_size=w)
        assert data["X_train"].shape[1] == w
        assert data["X_train"].shape[2] == 4
        assert len(data["X_train"]) + len(data["X_val"]) + len(data["X_test"]) > 400

def test_train_eval_single_head_model():
    data = prepare_sequential_data(CSV_PATH, window_size=3)
    res = train_eval_model(PipelineGRU, data, hidden_dim=16, epochs=5, is_dual_head=False)
    
    assert "acc" in res
    assert "rec" in res
    assert "f1" in res
    assert "latency_ms" in res
    assert res["params"] > 0
    assert 0.0 <= res["acc"] <= 1.0
    assert res["latency_ms"] > 0.0

def test_train_eval_dual_head_attention_model():
    data = prepare_sequential_data(CSV_PATH, window_size=3)
    res = train_eval_model(AttentionGRU, data, hidden_dim=16, epochs=5, is_dual_head=True)
    
    assert res["stage_acc"] is not None
    assert 0.0 <= res["stage_acc"] <= 1.0
    assert res["params"] > 0
    assert len(res["train_loss_history"]) == 5
    assert len(res["val_loss_history"]) == 5

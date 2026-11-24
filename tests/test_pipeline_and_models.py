"""
Unit tests for Data Pipeline Failure Prediction System (Week 1 End-Sem Deliverables).
Validates:
1. Relational linkage between pipeline_telemetry.csv and pipeline_logs.json (Primary Key / Foreign Key).
2. Data engineering pipeline telemetry features and label distributions.
3. Sequential sliding window generation (T=3, input_dim=4).
4. Deep learning model architectures (RNN, LSTM, GRU) forward pass.
5. Log analyzer root cause diagnostic engine.
"""

import os
import sys
import json
import pytest
import torch
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from src.features.sequence_generator import prepare_sequential_data
from src.models.rnn_model import PipelineRNN
from src.models.lstm_model import PipelineLSTM
from src.models.gru_model import PipelineGRU
from src.data.log_analyzer import load_logs_database, parse_log_root_cause, get_run_analysis_by_log_id

CSV_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "pipeline_telemetry.csv")
LOGS_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "pipeline_logs.json")

def test_telemetry_csv_and_logs_json_exist():
    assert os.path.exists(CSV_PATH), f"Telemetry CSV not found at {CSV_PATH}"
    assert os.path.exists(LOGS_PATH), f"Logs JSON not found at {LOGS_PATH}"

def test_relational_log_id_integrity():
    df = pd.read_csv(CSV_PATH)
    with open(LOGS_PATH, "r") as f:
        logs = json.load(f)

    assert len(df) == 450, f"Expected 450 telemetry runs, found {len(df)}"
    assert len(logs) == 450, f"Expected 450 log entries, found {len(logs)}"
    assert "log_id" in df.columns, "Foreign key 'log_id' column missing in telemetry CSV"

    # Verify every log_id in CSV exists in logs JSON
    csv_log_ids = set(df["log_id"].dropna().tolist())
    json_keys = set(logs.keys())
    missing_keys = csv_log_ids - json_keys
    assert len(missing_keys) == 0, f"Missing log entries in JSON for IDs: {missing_keys}"

def test_failure_classes_distribution():
    df = pd.read_csv(CSV_PATH)
    scenarios = set(df["failure_scenario"].unique())
    expected_scenarios = {
        "NONE", "API_TIMEOUT", "API_HTTP_500", "API_RATE_LIMIT_429", "API_MALFORMED_JSON",
        "S3_MISSING_KEY", "S3_ACCESS_DENIED", "S3_EMPTY_PAYLOAD", "SPARK_DATA_QUALITY_NULL",
        "SPARK_SCHEMA_DRIFT", "SPARK_OUT_OF_MEMORY", "POSTGRES_CONNECTION_REFUSED",
        "POSTGRES_POOL_EXHAUSTED", "POSTGRES_LOCK_TIMEOUT", "POSTGRES_DISK_FULL"
    }
    for scen in expected_scenarios:
        assert scen in scenarios, f"Scenario '{scen}' missing from telemetry dataset"

def test_log_analyzer_diagnostics():
    sample_scenarios = {
        "API_TIMEOUT": ("PRODUCER_API_EXTRACTION", "requests.exceptions.ReadTimeout"),
        "S3_ACCESS_DENIED": ("STORAGE_S3_STAGING", "java.nio.file.AccessDeniedException (403)"),
        "SPARK_DATA_QUALITY_NULL": ("SPARK_DISTRIBUTED_TRANSFORM", "ValueError (Data Quality Violation)"),
        "POSTGRES_CONNECTION_REFUSED": ("DATABASE_POSTGRES_LOAD", "psycopg2.OperationalError (Connection Refused)")
    }
    df = pd.read_csv(CSV_PATH)
    for scenario, (expected_stage, expected_err) in sample_scenarios.items():
        sample_row = df[df["failure_scenario"] == scenario].iloc[0]
        analysis = get_run_analysis_by_log_id(sample_row["log_id"], json_path=LOGS_PATH)
        assert analysis["status"] == "FAILED"
        assert analysis["stage"] == expected_stage, f"Failed for {scenario}: got {analysis['stage']}"
        assert analysis["error_signature"] == expected_err, f"Failed for {scenario}: got {analysis['error_signature']}"
        assert len(analysis["root_cause"]) > 10
        assert len(analysis["recommendation"]) > 10

def test_sequential_data_preparation():
    data = prepare_sequential_data(CSV_PATH, window_size=3)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]

    assert X_train.dim() == 3, "X_train must be 3-dimensional (Batch, Time, Features)"
    assert X_train.shape[1] == 3, "Sequence length must match window_size=3"
    assert X_train.shape[2] == 4, "Feature dimension must be 4"
    # Strict chronological partition without lookahead leakage
    assert len(X_train) == 312
    assert len(X_val) == 64
    assert len(X_test) == 65
    assert len(X_train) + len(X_val) + len(X_test) == 441

def test_model_forward_passes():
    batch_size = 8
    seq_len = 3
    feat_dim = 4
    dummy_input = torch.randn(batch_size, seq_len, feat_dim)

    models = [
        PipelineRNN(input_dim=feat_dim, hidden_dim=16),
        PipelineLSTM(input_dim=feat_dim, hidden_dim=16),
        PipelineGRU(input_dim=feat_dim, hidden_dim=16)
    ]

    for model in models:
        out = model(dummy_input)
        assert out.shape == (batch_size,), f"Shape mismatch for {model.__class__.__name__}"
        assert (out >= 0.0).all() and (out <= 1.0).all(), f"Output not in [0, 1] for {model.__class__.__name__}"

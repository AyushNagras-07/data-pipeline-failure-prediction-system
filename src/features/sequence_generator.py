"""
sequence_generator.py
Transforms tabular pipeline telemetry into 3D temporal sequence tensors (N, T, D)
using sliding windows of size T, strictly avoiding data leakage and performing
chronological Train / Validation / Test splits.
Includes log_id and run_id tracking to link sequential predictions directly to
the natural text logs stored in pipeline_logs.json, as well as multi-class stage targets
for dual-head sequential attention models.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch

FEATURE_COLUMNS = ['duration', 'records_processed', 'retry_count', 'memory_mb']
TARGET_COLUMN = 'status'

STAGE_MAPPING = {
    "NONE": 0,
    "API_TIMEOUT": 1,
    "API_HTTP_500": 1,
    "API_RATE_LIMIT_429": 1,
    "API_MALFORMED_JSON": 1,
    "S3_MISSING_KEY": 2,
    "S3_ACCESS_DENIED": 2,
    "S3_EMPTY_PAYLOAD": 2,
    "SPARK_DATA_QUALITY_NULL": 3,
    "SPARK_SCHEMA_DRIFT": 3,
    "SPARK_OUT_OF_MEMORY": 3,
    "POSTGRES_CONNECTION_REFUSED": 4,
    "POSTGRES_POOL_EXHAUSTED": 4,
    "POSTGRES_LOCK_TIMEOUT": 4,
    "POSTGRES_DISK_FULL": 4
}

STAGE_NAMES = [
    "HEALTHY",
    "API_INGESTION",
    "S3_STORAGE",
    "SPARK_PROCESSING",
    "POSTGRES_DATABASE"
]

def create_sliding_sequences(data_values, targets, metadata_ids, stage_targets=None, window_size=3):
    """
    Creates sequences of length `window_size` to predict the next run outcome.
    Past runs [t - window_size, ..., t - 1] -> Target at run t.
    Also tracks the log_id and run_id corresponding to target run t.
    If stage_targets is provided, returns multi-class stage targets as well.
    """
    X, y, target_ids, y_stage = [], [], [], []
    for i in range(window_size, len(data_values)):
        X.append(data_values[i - window_size : i])
        y.append(targets[i])
        target_ids.append(metadata_ids[i])
        if stage_targets is not None:
            y_stage.append(stage_targets[i])
            
    X_arr = np.array(X, dtype=np.float32)
    y_arr = np.array(y, dtype=np.float32)
    
    if stage_targets is not None:
        return X_arr, y_arr, target_ids, np.array(y_stage, dtype=np.int64)
    return X_arr, y_arr, target_ids

def prepare_sequential_data(csv_path, window_size=3, train_split=0.70, val_split=0.15):
    """
    Loads telemetry, splits chronologically, scales features without lookahead leakage,
    and returns PyTorch tensors for Train, Validation, and Test sets, alongside
    metadata identifiers (log_id, run_id) and multi-class stage labels.
    """
    df = pd.read_csv(csv_path)
    
    n_total = len(df)
    train_end = int(n_total * train_split)
    val_end = int(n_total * (train_split + val_split))
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    # Fit scaler ONLY on training data to avoid data leakage
    scaler = StandardScaler()
    scaler.fit(train_df[FEATURE_COLUMNS])
    
    train_features = scaler.transform(train_df[FEATURE_COLUMNS])
    val_features = scaler.transform(val_df[FEATURE_COLUMNS])
    test_features = scaler.transform(test_df[FEATURE_COLUMNS])
    
    train_targets = train_df[TARGET_COLUMN].values
    val_targets = val_df[TARGET_COLUMN].values
    test_targets = test_df[TARGET_COLUMN].values
    
    # Multi-class stage targets
    train_stage_targets = np.array([STAGE_MAPPING.get(s, 0) for s in train_df['failure_scenario']], dtype=np.int64)
    val_stage_targets = np.array([STAGE_MAPPING.get(s, 0) for s in val_df['failure_scenario']], dtype=np.int64)
    test_stage_targets = np.array([STAGE_MAPPING.get(s, 0) for s in test_df['failure_scenario']], dtype=np.int64)
    
    train_ids = train_df[['run_id', 'log_id', 'failure_scenario']].to_dict('records')
    val_ids = val_df[['run_id', 'log_id', 'failure_scenario']].to_dict('records')
    test_ids = test_df[['run_id', 'log_id', 'failure_scenario']].to_dict('records')
    
    # Generate sliding window sequences
    X_train, y_train, train_meta, y_stage_train = create_sliding_sequences(
        train_features, train_targets, train_ids, stage_targets=train_stage_targets, window_size=window_size
    )
    X_val, y_val, val_meta, y_stage_val = create_sliding_sequences(
        val_features, val_targets, val_ids, stage_targets=val_stage_targets, window_size=window_size
    )
    X_test, y_test, test_meta, y_stage_test = create_sliding_sequences(
        test_features, test_targets, test_ids, stage_targets=test_stage_targets, window_size=window_size
    )
    
    return {
        "X_train": torch.tensor(X_train),
        "y_train": torch.tensor(y_train),
        "y_stage_train": torch.tensor(y_stage_train, dtype=torch.long),
        "X_val": torch.tensor(X_val),
        "y_val": torch.tensor(y_val),
        "y_stage_val": torch.tensor(y_stage_val, dtype=torch.long),
        "X_test": torch.tensor(X_test),
        "y_test": torch.tensor(y_test),
        "y_stage_test": torch.tensor(y_stage_test, dtype=torch.long),
        "train_meta": train_meta,
        "val_meta": val_meta,
        "test_meta": test_meta,
        "scaler": scaler,
        "feature_names": FEATURE_COLUMNS,
        "stage_names": STAGE_NAMES,
        "stage_mapping": STAGE_MAPPING,
        "num_classes": len(STAGE_NAMES),
        "window_size": window_size,
        "total_runs": n_total
    }

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    csv_file = os.path.join(project_root, "data", "raw", "pipeline_telemetry.csv")
    data = prepare_sequential_data(csv_file, window_size=3)
    print(f"Total runs loaded: {data['total_runs']}")
    print(f"Sequence Generation Successful (Window Size = {data['window_size']}):")
    print(f"  • X_train shape: {data['X_train'].shape} | y_train shape: {data['y_train'].shape}")
    print(f"  • X_val shape:   {data['X_val'].shape} | y_val shape:   {data['y_val'].shape}")
    print(f"  • X_test shape:  {data['X_test'].shape} | y_test shape:  {data['y_test'].shape}")
    print(f"  • Sample Test Target Meta: {data['test_meta'][-1]}")

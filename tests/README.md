# Test Suite (`tests/`)

## 1. What This Folder Is
This directory will contain automated test suites (using frameworks such as `pytest`) to verify the correctness, reliability, and stability of our data engineering and deep learning code.

## 2. Why This Folder Exists
Data science and machine learning projects are prone to subtle bugs—such as silent tensor shape mismatches, inadvertent data leakage across sliding windows, and corrupted missing-value imputations. Automated tests ensure that:
- Preprocessing steps preserve data invariants.
- Sequence window generators produce mathematically sound shapes `(batch_size, time_steps, features)` without looking ahead into the test set.
- Model forward passes emit properly bounded probabilities $[0, 1]$.
- Metric calculations remain reproducible.

## 3. What Will Eventually Be Stored Here
When tests are implemented alongside future phases, this folder will contain:
- **`test_data.py`**: Unit tests for telemetry loading, schema validation, and missing data imputation in `src/data/`.
- **`test_features.py`**: Verification of feature transformations, scaling boundaries, and sliding-window sequence alignment in `src/features/`.
- **`test_models.py`**: Tensor shape verification and forward-pass integration tests for RNN, LSTM, and GRU models in `src/models/`.
- **`test_evaluation.py`**: Verification of classification metrics and confusion matrix calculations in `src/evaluation/`.

## 4. How It Connects to the Overall Project
Every module in `src/` will have corresponding tests in this directory. Running test commands will validate the entire workflow from raw data ingestion to sequential model inference.

## 5. What We Are NOT Doing Yet
- We are **not** writing test scripts or assertions at this stage.
- Test suites will be introduced incrementally alongside each module's implementation.

# Source Code (`src/`)

## 1. What This Folder Is
The `src/` directory is the core Python application package for the Data Pipeline Failure Prediction System.

## 2. Why This Folder Exists
Separating source code into modular, single-responsibility subpackages ensures code reusability, testability, and clarity. Instead of maintaining monolithic scripts or messy notebooks, each pipeline stage has a dedicated subpackage.

## 3. End-to-End Pipeline Flow
The code inside `src/` executes a streamlined, sequential flow:

```text
       ┌──────────────┐
       │   src/data   │  Data loading, validation, and cleaning
       └──────┬───────┘
              │
              ▼
     ┌─────────────────┐
     │  src/features   │  Feature extraction, scaling & temporal sequence creation
     └────────┬────────┘
              │
              ▼
       ┌──────────────┐
       │  src/models  │  Sequential architectures (RNN, LSTM, GRU) definition & training
       └──────┬───────┘
              │
              ▼
     ┌─────────────────┐
     │ src/evaluation  │  Performance evaluation, metric comparison & error analysis
     └─────────────────┘
```

## 4. Subpackages Overview
- **`src/data/`**: Ingests raw telemetry logs, executes schema validation, performs missing-value imputation, and saves cleaned data tables.
- **`src/features/`**: Computes health and rolling indicators, scales numeric signals, and converts tabular time series into 3D sequential sliding-window tensors `(samples, time_steps, features)`.
- **`src/models/`**: Defines, trains, and serializes the deep learning sequential models (Vanilla RNN, LSTM, and GRU).
- **`src/evaluation/`**: Evaluates trained models on holdout test splits, calculates classification metrics (Accuracy, Precision, Recall, F1, ROC-AUC), and produces comparison reports.

## 5. What We Are NOT Doing Yet
- We are **not** writing Python source scripts or model training loops at this stage.
- Subpackages currently contain architecture blueprints to guide future implementation phases.

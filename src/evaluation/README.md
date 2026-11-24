# Model Evaluation & Comparison (`src/evaluation/`)

## 1. What This Folder Is
This subpackage will contain the evaluation logic, performance metrics computation, comparative reporting tools, and error analysis scripts.

## 2. Why This Folder Exists
A core requirement of the ANNA subject and ESE project examination is a rigorous comparative analysis across architectures (RNN vs. LSTM vs. GRU). This directory provides dedicated, unbiased evaluation routines using identical test sets and evaluation protocols.

## 3. What Will Eventually Be Stored Here
This directory will contain:
- **`metrics.py`**: Implementation and calculation of classification metrics:
  - **Accuracy:** Overall proportion of correct predictions.
  - **Precision:** Of the runs predicted to fail, how many actually failed?
  - **Recall (Sensitivity):** Of the runs that actually failed, how many did the model catch?
  - **F1-Score:** Harmonic mean of precision and recall.
  - **ROC-AUC & PR-AUC:** Area under Receiver Operating Characteristic and Precision-Recall curves.
  - **Confusion Matrix:** Breakdown of True Positives, False Positives, True Negatives, and False Negatives.
- **`comparator.py`**: Comparative analysis utilities that generate comparative performance tables and charts across RNN, LSTM, and GRU models.
- **`error_analysis.py`**: Inspection of failure cases, misclassifications, and edge cases (e.g., sudden out-of-distribution pipeline crashes vs. gradual resource leaks).

### The Critical Importance of Recall and False Negatives
In an operational data pipeline failure-prediction system:
- A **False Positive** (model alerts high risk, but the pipeline succeeds) causes minor inconvenience (e.g., an engineer briefly checks logs or an automated resource buffer is allocated).
- A **False Negative** (model predicts success, but the pipeline crashes silently) causes severe real-world harm: delayed upstream dashboards, breached SLAs, corrupted downstream data warehouses, and urgent middle-of-the-night incident resolution.

Therefore, **Recall** (minimizing False Negatives) and the **Precision-Recall trade-off** will be treated as primary metrics alongside ROC-AUC, rather than relying solely on overall Accuracy.

## 4. How It Connects to the Overall Project
This subpackage receives model checkpoints and prediction probabilities from `src/models/`, evaluates them against ground-truth labels from `data/processed/`, and produces summary logs and figures that will be documented in `docs/`.

## 5. What We Are NOT Doing Yet
- We are **not** generating fake evaluation scores, confusion matrices, or benchmark tables.
- Evaluation scripts will be implemented during **Phase 7**.

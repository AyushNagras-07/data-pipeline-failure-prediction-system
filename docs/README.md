# Project Documentation (`docs/`)

## 1. What This Folder Is
This directory is the central repository for technical documentation, academic deliverables, experimental notes, and architectural design records.

## 2. Why This Folder Exists
As an academic project for **Advanced Neural Network Architectures (ANNA — Course Code: 2307411T)**, rigorous documentation is essential for both Mid-Review Evaluation (40 Marks) and End-Semester Project Examination (50 Marks). Maintaining structured documentation alongside code ensures all decisions, literature reviews, and empirical results are captured systematically.

## 3. What Will Eventually Be Stored Here
As the project progresses through future phases, this folder will house:
- **`architecture_decisions.md`**: Technical justifications for design choices (e.g., choice of sequential models, sequence window lengths, feature sets).
- **`literature_survey_and_gap.md`**: Summaries of reviewed research papers on pipeline failure prediction and time-series anomaly detection, highlighting the specific research gap addressed.
- **`methodology.md`**: Detailed explanations of telemetry preprocessing, sequence generation, loss formulations, and optimization strategies (learning rates, optimizers, regularization).
- **`experiment_notes.md`**: Logs of training runs, hyperparameter sweeps, convergence curves, and optimization challenges.
- **`model_comparison.md`**: Comparative performance tables and trade-off analyses (RNN vs. LSTM vs. GRU on Accuracy, Precision, Recall, F1, and computational latency).
- **`diagrams/`**: High-level workflow charts, neural network architecture illustrations, and visual evaluation results (confusion matrices, ROC curves).

## 4. How It Connects to the Overall Project
This folder synthesizes the engineering outputs from `src/` and experimental results from `src/evaluation/` into formal academic documentation, serving as the basis for the technical project report, viva defense, and research paper presentation.

## 5. What We Are NOT Doing Yet
- We are **not** creating premature or boilerplate documentation files at this stage.
- Documentation will be authored progressively as empirical experiments and modules are completed.

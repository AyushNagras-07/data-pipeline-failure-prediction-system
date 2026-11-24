# Airflow Directed Acyclic Graphs (`airflow/dags/`)

## 1. What This Folder Is
This directory will house the Apache Airflow DAG (Directed Acyclic Graph) Python definitions.

## 2. Why This Folder Exists
In Apache Airflow, DAG files specify the workflow structure, task dependencies, schedules, and retry policies. Having a dedicated `dags/` folder allows Airflow to discover and execute workflow definitions cleanly.

## 3. What Will Eventually Be Stored Here
This directory will eventually contain:
- Controlled, multi-stage data engineering DAGs (e.g., Extract, Transform, Validate, Load pipelines).
- Configurable scenarios that introduce realistic pipeline variations (varying input record volumes, simulated network timeouts, transient database locks, and computational spikes) to produce a rich and balanced telemetry dataset containing both successful runs and failure modes.

## 4. How It Connects to the Overall Project
The DAGs defined here run on the scheduler and generate concrete execution instances. Telemetry extracted from these executions forms the raw dataset stored in `data/raw/`, powering the sequential models trained to predict upcoming run failures.

## 5. What We Are NOT Doing Yet
- We are **not** writing any DAG Python scripts at this stage.
- DAG definitions will be created in **Phase 2** when establishing the telemetry dataset.

# Apache Airflow Integration (`airflow/`)

## 1. What This Folder Is
This directory is reserved for Apache Airflow configuration files, plugins, and pipeline definitions used to generate or simulate real-world data pipeline workloads.

## 2. Why Airflow Is Relevant to This Project
Apache Airflow is an industry-standard workflow orchestration platform used by modern data engineering teams to manage directed acyclic graphs (DAGs) of tasks. 

In our project:
- Airflow serves as the **target system and source environment** whose operational behavior we monitor and predict.
- As Airflow executes pipeline DAGs, its metadata database and task logs record rich execution telemetry, including:
  - Task execution state (`success`, `failed`, `up_for_retry`, `queued`)
  - Start, end, and elapsed execution durations
  - Task retry counts and timeout events
  - Task queue and worker scheduling delays
  - Worker resource utilization and system failure signals

Rather than predicting abstract, synthetic numbers, Airflow grounds our project in real-world data engineering workflows.

## 3. What Will Eventually Be Stored Here
When Airflow integration is established in a later phase, this folder will contain:
- `airflow.cfg` / environment configurations.
- Custom execution loggers or telemetry export hooks.
- Pipeline task definitions located within `airflow/dags/`.

## 4. How It Connects to the Overall Project
Pipelines defined and triggered through Airflow generate execution records. A lightweight telemetry extraction hook or log parser reads these events, writing raw operational metrics into `data/raw/` for our prediction pipeline.

## 5. What We Are NOT Doing Yet
- We are **not** installing, spinning up, or configuring an Airflow instance at this stage.
- We are **not** creating background workers, databases, or Docker containers.
- Airflow environment setup and DAG creation will be handled in **Phase 2** when telemetry generation is needed.

"""
generate_scaled_telemetry.py
Generates a realistic, production-scale dataset of 450 historical pipeline runs
spanning a 15-month operational window.

Features:
- 15 realistic failure classes covering Ingestion, Storage, Spark, and PostgreSQL.
- Compounding multi-run degradation sequences (memory leaks, creeping latency, retries).
- Generates pipeline_telemetry.csv (metrics + foreign key log_id).
- Generates pipeline_logs.json (natural Airflow task logs keyed by log_id).
"""

import os
import json
import random
import pandas as pd
from datetime import datetime, timedelta

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
CSV_OUTPUT = os.path.join(RAW_DIR, "pipeline_telemetry.csv")
LOGS_JSON_OUTPUT = os.path.join(RAW_DIR, "pipeline_logs.json")

def generate_natural_log(run_id, log_id, dt_str, scenario, duration, records, memory_mb, retries):
    """
    Constructs an authentic, natural-looking multi-task Airflow execution log.
    Includes timestamps, subtask lifecycles, and exact Python/PySpark/Postgres tracebacks.
    """
    ts = dt_str.replace(" ", "T") + "+0000"
    
    if scenario == "NONE":
        log_text = f"""*** Log file for DAG: producer_users_daily / consumer_users_daily | Run ID: {run_id} | Log ID: {log_id}
[{ts}] {{taskinstance.py:2076}} INFO - Dependencies all met for ti=<TaskInstance: producer_users_daily.producer {run_id} [queued]>
[{ts}] {{taskinstance.py:2306}} INFO - Starting attempt 1 of 6
[{ts}] {{standard_task_runner.py:63}} INFO - Started process 412 to run task
[{ts}] {{producer_users_daily.py:21}} INFO - Calling JSONPlaceholder API: https://jsonplaceholder.typicode.com/users
[{ts}] {{producer_users_daily.py:23}} INFO - HTTP GET returned status 200 OK. Extracted {records} records.
[{ts}] {{s3_hook.py:145}} INFO - Uploading data/sample/{dt_str[:10]}.json to S3 bucket ayush-consumer-producer (replace=True)
[{ts}] {{taskinstance.py:1150}} INFO - Task producer_users_daily.producer completed with state SUCCESS
[{ts}] {{external_task_sensor.py:112}} INFO - External task producer_users_daily.producer finished in state SUCCESS.
[{ts}] {{s3_sensor.py:88}} INFO - Poking for bucket: ayush-consumer-producer, key: data/sample/{dt_str[:10]}.json
[{ts}] {{s3_sensor.py:95}} INFO - Key data/sample/{dt_str[:10]}.json found in S3.
[{ts}] {{consumer_users_daily.py:25}} INFO - Initializing SparkSession: appName='consumer-transform', master='local[*]'
[{ts}] {{consumer_users_daily.py:46}} INFO - Loaded {records} records from s3a://ayush-consumer-producer/data/sample/{dt_str[:10]}.json
[{ts}] {{consumer_users_daily.py:53}} INFO - Running Data Quality Check: Null count on id and email = 0. PASS.
[{ts}] {{consumer_users_daily.py:67}} INFO - Writing clean Parquet partition to s3a://ayush-consumer-producer/data/processed/users/ds={dt_str[:10]}/
[{ts}] {{postgres_hook.py:82}} INFO - Connecting to PostgreSQL database 'airflow' via postgres_default
[{ts}] {{consumer_users_daily.py:100}} INFO - Executing: DELETE FROM users_daily WHERE dt = '{dt_str[:10]}'
[{ts}] {{consumer_users_daily.py:108}} INFO - Idempotent partition load complete. Inserted {records} rows. Transaction committed.
[{ts}] {{taskinstance.py:1150}} INFO - Task consumer_users_daily.load_to_postgres completed with state SUCCESS
[{ts}] {{local_task_job_runner.py:228}} INFO - Run duration: {duration}s | Resident Memory Peak: {memory_mb} MB | Overall Status: SUCCESS"""

    elif scenario == "API_TIMEOUT":
        log_text = f"""*** Log file for DAG: producer_users_daily | Run ID: {run_id} | Log ID: {log_id}
[{ts}] {{taskinstance.py:2306}} INFO - Starting attempt {retries} of 6
[{ts}] {{producer_users_daily.py:21}} INFO - Calling JSONPlaceholder API: https://jsonplaceholder.typicode.com/users
[{ts}] {{connectionpool.py:715}} WARNING - Retrying (Retry(total=2, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ReadTimeoutError'
[{ts}] {{taskinstance.py:2905}} ERROR - Task failed with exception
Traceback (most recent call last):
  File "/opt/airflow/dags/producer_users_daily.py", line 20, in api_data
    resp = requests.get("https://jsonplaceholder.typicode.com/users", timeout=10)
  File "/home/airflow/.local/lib/python3.10/site-packages/requests/api.py", line 73, in get
    return request('get', url, params=params, **kwargs)
requests.exceptions.ReadTimeout: HTTPSConnectionPool(host='jsonplaceholder.typicode.com', port=443): Read timed out. (read timeout=10)
[{ts}] {{taskinstance.py:1150}} ERROR - Task producer_users_daily.producer failed after {retries} retry attempts."""

    elif scenario == "API_HTTP_500":
        log_text = f"""*** Log file for DAG: producer_users_daily | Run ID: {run_id} | Log ID: {log_id}
[{ts}] {{taskinstance.py:2306}} INFO - Starting attempt {retries} of 6
[{ts}] {{producer_users_daily.py:21}} INFO - Calling JSONPlaceholder API: https://jsonplaceholder.typicode.com/users
[{ts}] {{taskinstance.py:2905}} ERROR - Task failed with exception
Traceback (most recent call last):
  File "/opt/airflow/dags/producer_users_daily.py", line 22, in api_data
    resp.raise_for_status()
  File "/home/airflow/.local/lib/python3.10/site-packages/requests/models.py", line 1021, in raise_for_status
    raise HTTPError(http_error_msg, response=self)
requests.exceptions.HTTPError: 500 Server Error: Internal Server Error for url: https://jsonplaceholder.typicode.com/users"""

    elif scenario == "API_RATE_LIMIT_429":
        log_text = f"""*** Log file for DAG: producer_users_daily | Run ID: {run_id} | Log ID: {log_id}
[{ts}] {{taskinstance.py:2306}} INFO - Starting attempt {retries} of 6
[{ts}] {{producer_users_daily.py:20}} INFO - Calling JSONPlaceholder API: https://jsonplaceholder.typicode.com/users
[{ts}] {{taskinstance.py:2905}} ERROR - Task failed with exception
Traceback (most recent call last):
  File "/opt/airflow/dags/producer_users_daily.py", line 22, in api_data
    resp.raise_for_status()
requests.exceptions.HTTPError: 429 Client Error: Too Many Requests. API quota exceeded for IP."""

    elif scenario == "API_MALFORMED_JSON":
        log_text = f"""*** Log file for DAG: producer_users_daily | Run ID: {run_id} | Log ID: {log_id}
[{ts}] {{taskinstance.py:2306}} INFO - Starting attempt 1 of 6
[{ts}] {{producer_users_daily.py:20}} INFO - Calling API endpoint... Received response payload.
[{ts}] {{taskinstance.py:2905}} ERROR - Task failed with exception
Traceback (most recent call last):
  File "/opt/airflow/dags/producer_users_daily.py", line 23, in api_data
    return resp.json()
  File "/usr/lib/python3.10/json/decoder.py", line 337, in decode
    raise JSONDecodeError("Expecting value", s, err.value) from None
json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0). Upstream provider returned invalid HTML 502 Bad Gateway text."""

    elif scenario == "S3_MISSING_KEY":
        log_text = f"""*** Log file for DAG: consumer_users_daily | Run ID: {run_id} | Log ID: {log_id}
[{ts}] {{s3_sensor.py:88}} INFO - Poking for bucket: ayush-consumer-producer, key: data/sample/{dt_str[:10]}.json
[{ts}] {{s3_sensor.py:102}} INFO - Key data/sample/{dt_str[:10]}.json does not exist. Rescheduling in 60s.
[{ts}] {{taskinstance.py:2905}} ERROR - Task failed with exception
Traceback (most recent call last):
  File "/opt/airflow/dags/consumer_users_daily.py", line 142, in wait_for_s3_file
    raise AirflowSensorTimeout('Sensor has timed out.')
airflow.exceptions.AirflowSensorTimeout: Sensor wait_for_s3_file timed out after 30000.0 seconds. Upstream producer did not stage partition."""

    elif scenario == "S3_ACCESS_DENIED":
        log_text = f"""*** Log file for DAG: consumer_users_daily | Run ID: {run_id} | Log ID: {log_id}
[{ts}] {{taskinstance.py:2306}} INFO - Starting attempt 1 of 3
[{ts}] {{consumer_users_daily.py:25}} INFO - Initializing SparkSession...
[{ts}] {{taskinstance.py:2905}} ERROR - Task failed with exception
Traceback (most recent call last):
  File "/opt/airflow/dags/consumer_users_daily.py", line 46, in transform_users
    df = spark.read.schema(schema).json(raw_path)
py4j.protocol.Py4JJavaError: An error occurred while calling o43.json.
: java.nio.file.AccessDeniedException: s3a://ayush-consumer-producer/data/sample/{dt_str[:10]}.json: com.amazonaws.services.s3.model.AmazonS3Exception: Forbidden (Status Code: 403; Error Code: 403 Forbidden). IAM role policy invalid."""

    elif scenario == "S3_EMPTY_PAYLOAD":
        log_text = f"""*** Log file for DAG: consumer_users_daily | Run ID: {run_id} | Log ID: {log_id}
[{ts}] {{taskinstance.py:2306}} INFO - Starting attempt 1 of 3
[{ts}] {{s3_sensor.py:95}} INFO - Key data/sample/{dt_str[:10]}.json found in S3.
[{ts}] {{consumer_users_daily.py:46}} INFO - Ingesting raw JSON into Spark...
[{ts}] {{taskinstance.py:2905}} ERROR - Task failed with exception
py4j.protocol.Py4JJavaError: An error occurred while calling o50.json.
: org.apache.spark.sql.AnalysisException: Since the input path doesn't contain any data, it cannot be read as JSON. File size is 0 bytes."""

    elif scenario == "SPARK_DATA_QUALITY_NULL":
        log_text = f"""*** Log file for DAG: consumer_users_daily | Run ID: {run_id} | Log ID: {log_id}
[{ts}] {{consumer_users_daily.py:48}} INFO - Total records ingested from S3: 10
[{ts}] {{consumer_users_daily.py:51}} INFO - Checking identity columns for null values...
[{ts}] {{taskinstance.py:2905}} ERROR - Task failed with exception
Traceback (most recent call last):
  File "/opt/airflow/dags/consumer_users_daily.py", line 56, in transform_users
    raise ValueError(f"Data quality check failed. {{null_count}} rows contain null values.")
ValueError: Data quality check failed. 3 rows contain null values in required columns 'id' and 'email'."""

    elif scenario == "SPARK_SCHEMA_DRIFT":
        log_text = f"""*** Log file for DAG: consumer_users_daily | Run ID: {run_id} | Log ID: {log_id}
[{ts}] {{consumer_users_daily.py:46}} INFO - Applying predefined StructType schema...
[{ts}] {{taskinstance.py:2905}} ERROR - Task failed with exception
py4j.protocol.Py4JJavaError: An error occurred while calling o65.json.
: org.apache.spark.sql.AnalysisException: Cannot cast string field 'id' with value 'USR_9801_ABC' to IntegerType. Incompatible schema drift detected in source payload."""

    elif scenario == "SPARK_OUT_OF_MEMORY":
        log_text = f"""*** Log file for DAG: consumer_users_daily | Run ID: {run_id} | Log ID: {log_id}
[{ts}] {{consumer_users_daily.py:25}} INFO - Launching PySpark distributed executor on local JVM.
[{ts}] {{taskinstance.py:2905}} ERROR - Task failed with exception
py4j.protocol.Py4JJavaError: An error occurred while executing Spark action.
: java.lang.OutOfMemoryError: Java heap space. Resident set size exceeded JVM heap limit (Worker Memory: {memory_mb} MB). GC overhead limit exceeded."""

    elif scenario == "POSTGRES_CONNECTION_REFUSED":
        log_text = f"""*** Log file for DAG: consumer_users_daily | Run ID: {run_id} | Log ID: {log_id}
[{ts}] {{postgres_hook.py:82}} INFO - Attempting connection to postgres_default on host 'localhost' port 5432
[{ts}] {{taskinstance.py:2905}} ERROR - Task failed with exception
Traceback (most recent call last):
  File "/opt/airflow/dags/consumer_users_daily.py", line 97, in load_to_postgres
    conn = pg_hook.get_conn()
psycopg2.OperationalError: could not connect to server: Connection refused. Is the server running on host "localhost" (127.0.0.1) and accepting TCP/IP connections on port 5432?"""

    elif scenario == "POSTGRES_POOL_EXHAUSTED":
        log_text = f"""*** Log file for DAG: consumer_users_daily | Run ID: {run_id} | Log ID: {log_id}
[{ts}] {{postgres_hook.py:82}} INFO - Requesting connection handle from connection pool...
[{ts}] {{taskinstance.py:2905}} ERROR - Task failed with exception
Traceback (most recent call last):
  File "/opt/airflow/dags/consumer_users_daily.py", line 97, in load_to_postgres
    conn = pg_hook.get_conn()
psycopg2.OperationalError: FATAL: remaining connection slots are reserved for non-replication superuser connections (max_connections=100 reached)."""

    elif scenario == "POSTGRES_LOCK_TIMEOUT":
        log_text = f"""*** Log file for DAG: consumer_users_daily | Run ID: {run_id} | Log ID: {log_id}
[{ts}] {{consumer_users_daily.py:100}} INFO - Executing: DELETE FROM users_daily WHERE dt = '{dt_str[:10]}'
[{ts}] {{taskinstance.py:2905}} ERROR - Task failed with exception
Traceback (most recent call last):
  File "/opt/airflow/dags/consumer_users_daily.py", line 100, in load_to_postgres
    cursor.execute("DELETE FROM users_daily WHERE dt = %s", (ds,))
psycopg2.errors.LockNotAvailable: canceling statement due to lock timeout. Deadlock detected waiting for ExclusiveLock on table 'users_daily'."""

    elif scenario == "POSTGRES_DISK_FULL":
        log_text = f"""*** Log file for DAG: consumer_users_daily | Run ID: {run_id} | Log ID: {log_id}
[{ts}] {{consumer_users_daily.py:103}} INFO - Inserting transformed records into users_daily...
[{ts}] {{taskinstance.py:2905}} ERROR - Task failed with exception
Traceback (most recent call last):
  File "/opt/airflow/dags/consumer_users_daily.py", line 108, in load_to_postgres
    conn.commit()
psycopg2.errors.DiskFull: could not extend file "base/16384/24576": No space left on device. PostgreSQL tablespace exhausted."""

    else:
        log_text = f"[{ts}] INFO - Generic log for {run_id} ({log_id})."

    return log_text

def generate_scaled_telemetry_dataset(num_runs=450):
    random.seed(42)
    start_dt = datetime(2025, 8, 1, 6, 0, 0)
    
    # 15 failure categories
    failure_scenarios = [
        "API_TIMEOUT", "API_HTTP_500", "API_RATE_LIMIT_429", "API_MALFORMED_JSON",
        "S3_MISSING_KEY", "S3_ACCESS_DENIED", "S3_EMPTY_PAYLOAD",
        "SPARK_DATA_QUALITY_NULL", "SPARK_SCHEMA_DRIFT", "SPARK_OUT_OF_MEMORY",
        "POSTGRES_CONNECTION_REFUSED", "POSTGRES_POOL_EXHAUSTED", "POSTGRES_LOCK_TIMEOUT", "POSTGRES_DISK_FULL"
    ]
    
    records = []
    logs_dict = {}
    
    current_dt = start_dt
    base_memory = 84.0
    
    # Track temporal degradation state across runs
    degradation_counter = 0
    degradation_type = None
    
    # Round-robin index ensuring all 14 failure classes appear evenly
    failure_idx = 0
    
    for i in range(1, num_runs + 1):
        run_id = f"run_{current_dt.strftime('%Y%m%d')}_{i:03d}"
        log_id = f"log_{current_dt.strftime('%Y%m%d')}_{i:03d}"
        dt_str = current_dt.strftime("%Y-%m-%d %H:%M:%S")
        
        # Decide if this run starts or continues a degradation sequence
        # We aim for ~16% total failure rate (approx 72 failures out of 450 runs)
        if degradation_counter == 0 and random.random() < 0.14:
            # Start a 3-run degradation sequence leading to a crash
            degradation_counter = 3
            degradation_type = failure_scenarios[failure_idx % len(failure_scenarios)]
            failure_idx += 1
            
        if degradation_counter > 0:
            if degradation_counter == 3:
                # Run 1 of degradation (subtle early warning)
                duration = round(random.uniform(3.2, 5.0), 2)
                records_processed = 10
                retry_count = 1 if "API" in degradation_type else 0
                memory_mb = round(base_memory + random.uniform(10, 20), 2)
                scenario = "NONE"
                status = 0
                degradation_counter -= 1
            elif degradation_counter == 2:
                # Run 2 of degradation (strong pre-failure indicator)
                duration = round(random.uniform(5.5, 9.0), 2)
                records_processed = 10
                retry_count = random.choice([1, 2])
                memory_mb = round(base_memory + random.uniform(30, 55), 2)
                scenario = "NONE"
                status = 0
                degradation_counter -= 1
            elif degradation_counter == 1:
                # Run 3 of degradation: THE ACTUAL FAILURE
                scenario = degradation_type
                status = 1
                if "TIMEOUT" in scenario or "SENSOR" in scenario or "S3_MISSING" in scenario:
                    duration = 30.0 if "S3" in scenario else 10.0
                    records_processed = 0
                    retry_count = 5 if "API" in scenario else 3
                    memory_mb = round(base_memory + 15.0, 2)
                elif "MEMORY" in scenario:
                    duration = round(random.uniform(12.0, 20.0), 2)
                    records_processed = 0
                    retry_count = 0
                    memory_mb = round(base_memory + random.uniform(120, 180), 2) # Memory spike
                elif "DATA_QUALITY" in scenario:
                    duration = round(random.uniform(1.8, 2.8), 2)
                    records_processed = 0
                    retry_count = 0
                    memory_mb = round(base_memory + 25.0, 2)
                else:
                    duration = round(random.uniform(2.5, 6.0), 2)
                    records_processed = 0
                    retry_count = random.choice([1, 2, 3])
                    memory_mb = round(base_memory + 20.0, 2)
                degradation_counter = 0
                degradation_type = None
        else:
            # Healthy normal run
            duration = round(random.uniform(1.6, 2.7), 2)
            records_processed = 10
            retry_count = 0
            memory_mb = round(base_memory + random.uniform(-3, 8), 2)
            scenario = "NONE"
            status = 0
            
        records.append({
            "run_id": run_id,
            "timestamp": dt_str,
            "duration": duration,
            "records_processed": records_processed,
            "retry_count": retry_count,
            "memory_mb": memory_mb,
            "failure_scenario": scenario,
            "status": status,
            "log_id": log_id
        })
        
        # Generate the natural Airflow task log text for this run
        log_text = generate_natural_log(run_id, log_id, dt_str, scenario, duration, records_processed, memory_mb, retry_count)
        logs_dict[log_id] = log_text
        
        current_dt += timedelta(days=1, hours=random.randint(0, 1))

    df = pd.DataFrame(records)
    os.makedirs(RAW_DIR, exist_ok=True)
    
    # Save CSV
    df.to_csv(CSV_OUTPUT, index=False)
    
    # Save JSON logs
    with open(LOGS_JSON_OUTPUT, "w") as f:
        json.dump(logs_dict, f, indent=2)
        
    print(f"Generated {len(df)} runs in {CSV_OUTPUT}")
    print(f"Generated {len(logs_dict)} natural Airflow task logs in {LOGS_JSON_OUTPUT}")
    print(f"Class breakdown: Healthy runs = {(df['status'] == 0).sum()} | Failed runs = {(df['status'] == 1).sum()}")
    print("\nFailure Scenario counts:")
    print(df['failure_scenario'].value_counts())
    return df, logs_dict

if __name__ == "__main__":
    generate_scaled_telemetry_dataset()

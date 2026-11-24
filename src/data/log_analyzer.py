"""
log_analyzer.py
Engine for parsing and analyzing raw Airflow execution logs.
Connects the primary key 'log_id' from pipeline_telemetry.csv to the
natural text logs stored in pipeline_logs.json, extracting the root cause,
error signatures, and recommended engineer actions.
"""

import os
import json
import re

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
LOGS_JSON_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "pipeline_logs.json")

def load_logs_database(json_path=LOGS_JSON_PATH):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Logs database not found at {json_path}")
    with open(json_path, "r") as f:
        return json.load(f)

def parse_log_root_cause(log_text):
    """
    Parses a natural Airflow task execution log and identifies:
    - Pipeline Stage
    - Error Signature (Exception class)
    - Root Cause Explanation
    - Actionable Engineering Recommendation
    """
    if "Overall Status: SUCCESS" in log_text or "Task consumer_users_daily.load_to_postgres completed with state SUCCESS" in log_text:
        return {
            "status": "SUCCESS",
            "stage": "COMPLETE",
            "error_signature": None,
            "root_cause": "All pipeline tasks completed successfully within SLA.",
            "recommendation": "No action required. Normal operational state."
        }

    # Pattern 1: API Timeout
    if "ReadTimeout" in log_text or "ConnectTimeout" in log_text:
        return {
            "status": "FAILED",
            "stage": "PRODUCER_API_EXTRACTION",
            "error_signature": "requests.exceptions.ReadTimeout",
            "root_cause": "Upstream REST API took >10s to respond; connection pool timed out.",
            "recommendation": "Increase request timeout parameter or verify upstream API service status."
        }

    # Pattern 2: API 500
    if "500 Server Error" in log_text:
        return {
            "status": "FAILED",
            "stage": "PRODUCER_API_EXTRACTION",
            "error_signature": "requests.exceptions.HTTPError (500)",
            "root_cause": "Upstream source server returned HTTP 500 Internal Server Error.",
            "recommendation": "Notify API provider of backend server crash; engage backoff retry."
        }

    # Pattern 3: API 429 Rate Limit
    if "429 Client Error" in log_text or "Too Many Requests" in log_text:
        return {
            "status": "FAILED",
            "stage": "PRODUCER_API_EXTRACTION",
            "error_signature": "requests.exceptions.HTTPError (429)",
            "root_cause": "API request rate limit exceeded daily quota for the ingestion IP.",
            "recommendation": "Throttle request frequency or request rate-limit quota increase."
        }

    # Pattern 4: Malformed JSON
    if "JSONDecodeError" in log_text:
        return {
            "status": "FAILED",
            "stage": "PRODUCER_API_EXTRACTION",
            "error_signature": "json.decoder.JSONDecodeError",
            "root_cause": "Source returned invalid HTML gateway text instead of expected JSON array.",
            "recommendation": "Add upstream payload format validation and defensive parsing."
        }

    # Pattern 5: S3 Key Missing (Sensor Timeout)
    if "AirflowSensorTimeout" in log_text:
        return {
            "status": "FAILED",
            "stage": "STORAGE_S3_STAGING",
            "error_signature": "airflow.exceptions.AirflowSensorTimeout",
            "root_cause": "S3KeySensor timed out waiting for raw JSON partition in data lake.",
            "recommendation": "Inspect upstream Producer DAG run history to check if upload was skipped."
        }

    # Pattern 6: S3 Access Denied (403)
    if "403 Forbidden" in log_text or "AccessDeniedException" in log_text:
        return {
            "status": "FAILED",
            "stage": "STORAGE_S3_STAGING",
            "error_signature": "java.nio.file.AccessDeniedException (403)",
            "root_cause": "IAM role credentials expired or S3 bucket policy rejected read access.",
            "recommendation": "Refresh AWS session credentials and check IAM S3 policy permissions."
        }

    # Pattern 7: S3 Empty Payload
    if "File size is 0 bytes" in log_text or "doesn't contain any data" in log_text:
        return {
            "status": "FAILED",
            "stage": "STORAGE_S3_STAGING",
            "error_signature": "org.apache.spark.sql.AnalysisException (Empty)",
            "root_cause": "Staged S3 object exists but has 0 bytes (empty file anomaly).",
            "recommendation": "Check upstream producer file serialization before committing upload."
        }

    # Pattern 8: Spark Data Quality Null Violation
    if "Data quality check failed" in log_text:
        return {
            "status": "FAILED",
            "stage": "SPARK_DISTRIBUTED_TRANSFORM",
            "error_signature": "ValueError (Data Quality Violation)",
            "root_cause": "PySpark schema filter caught null records in identity columns 'id'/'email'.",
            "recommendation": "Route corrupt records to dead-letter quarantine queue and alert provider."
        }

    # Pattern 9: Spark Schema Drift
    if "Incompatible schema drift" in log_text or "Cannot cast" in log_text:
        return {
            "status": "FAILED",
            "stage": "SPARK_DISTRIBUTED_TRANSFORM",
            "error_signature": "org.apache.spark.sql.AnalysisException (Schema Drift)",
            "root_cause": "Source schema altered column types without backward compatibility.",
            "recommendation": "Update PySpark StructType schema to accommodate new source column types."
        }

    # Pattern 10: Spark Out of Memory
    if "OutOfMemoryError" in log_text or "Java heap space" in log_text:
        return {
            "status": "FAILED",
            "stage": "SPARK_DISTRIBUTED_TRANSFORM",
            "error_signature": "java.lang.OutOfMemoryError (Heap Space)",
            "root_cause": "PySpark driver/executor heap exhausted during distributed partition transform.",
            "recommendation": "Increase spark.driver.memory from 1G to 4G or enable partition repartitioning."
        }

    # Pattern 11: Postgres Connection Refused
    if "Connection refused" in log_text:
        return {
            "status": "FAILED",
            "stage": "DATABASE_POSTGRES_LOAD",
            "error_signature": "psycopg2.OperationalError (Connection Refused)",
            "root_cause": "Target PostgreSQL database daemon is down or port 5432 is unreachable.",
            "recommendation": "Verify database container status and check network firewall rules."
        }

    # Pattern 12: Postgres Connection Pool Exhaustion
    if "remaining connection slots are reserved" in log_text:
        return {
            "status": "FAILED",
            "stage": "DATABASE_POSTGRES_LOAD",
            "error_signature": "psycopg2.OperationalError (Pool Exhausted)",
            "root_cause": "PostgreSQL connection pool max_connections limit reached.",
            "recommendation": "Deploy PgBouncer connection pooler or increase max_connections in postgresql.conf."
        }

    # Pattern 13: Postgres Deadlock / Lock Timeout
    if "LockNotAvailable" in log_text or "lock timeout" in log_text:
        return {
            "status": "FAILED",
            "stage": "DATABASE_POSTGRES_LOAD",
            "error_signature": "psycopg2.errors.LockNotAvailable (Lock Timeout)",
            "root_cause": "Deadlock encountered during partition delete-and-insert operation.",
            "recommendation": "Tune transaction isolation level or execute partition drop concurrently."
        }

    # Pattern 14: Postgres Disk Full
    if "DiskFull" in log_text or "No space left on device" in log_text:
        return {
            "status": "FAILED",
            "stage": "DATABASE_POSTGRES_LOAD",
            "error_signature": "psycopg2.errors.DiskFull",
            "root_cause": "Storage volume hosting PostgreSQL tablespace has 0 bytes free.",
            "recommendation": "Expand storage volume or purge old historical partitions."
        }

    return {
        "status": "FAILED",
        "stage": "UNKNOWN",
        "error_signature": "SystemException",
        "root_cause": "Unclassified task failure encountered.",
        "recommendation": "Inspect full raw task trace in Airflow logs directory."
    }

def get_run_analysis_by_log_id(log_id, json_path=LOGS_JSON_PATH):
    logs_db = load_logs_database(json_path)
    if log_id not in logs_db:
        return {"error": f"Log ID '{log_id}' not found in logs database."}
    raw_log = logs_db[log_id]
    analysis = parse_log_root_cause(raw_log)
    analysis["log_id"] = log_id
    analysis["raw_log"] = raw_log
    return analysis

if __name__ == "__main__":
    logs_db = load_logs_database()
    sample_key = list(logs_db.keys())[0]
    print(f"Loaded {len(logs_db)} logs. Analyzing sample key '{sample_key}':")
    res = get_run_analysis_by_log_id(sample_key)
    print(json.dumps({k: v for k, v in res.items() if k != "raw_log"}, indent=2))

"""
generate_telemetry_history.py
Generates a realistic, chronological history of 75 pipeline runs based on the
Producer-Consumer ETL pipeline with normal runs and the 3 controlled failure scenarios:
1. API_TIMEOUT
2. MISSING_S3_FILE
3. DATA_QUALITY_NULL
"""

import os
import random
import pandas as pd
from datetime import datetime, timedelta

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
OUTPUT_CSV = os.path.join(PROJECT_ROOT, "data", "raw", "pipeline_telemetry.csv")

def generate_telemetry_dataset(num_runs=75):
    random.seed(42)
    start_dt = datetime(2026, 8, 15, 6, 0, 0)
    
    records = []
    
    # Base states
    current_dt = start_dt
    base_mem = 82.0
    
    for i in range(1, num_runs + 1):
        run_id = f"run_{current_dt.strftime('%Y%m%d')}_{i:03d}"
        timestamp = current_dt.strftime("%Y-%m-%d %H:%M:%S")
        
        # Determine failure scenario pattern (roughly 20% failure rate)
        # We introduce realistic degradation before failures so sequential models have patterns to learn!
        if i in [12, 13, 14]:  # Sequence leading to API Timeout
            if i < 14:
                # Degrading network response
                duration = round(random.uniform(3.5, 5.8), 2)
                records_processed = 10
                retry_count = 1
                memory_mb = round(base_mem + random.uniform(5, 12), 2)
                scenario = "NONE"
                status = 0
            else:
                # Hard API Timeout failure
                duration = 10.0
                records_processed = 0
                retry_count = 5
                memory_mb = round(base_mem + random.uniform(2, 6), 2)
                scenario = "API_TIMEOUT"
                status = 1
                
        elif i in [28, 29, 30]:  # Sequence leading to Missing S3 File
            if i < 30:
                duration = round(random.uniform(4.0, 7.5), 2)
                records_processed = 10
                retry_count = random.choice([0, 1])
                memory_mb = round(base_mem + random.uniform(10, 20), 2)
                scenario = "NONE"
                status = 0
            else:
                duration = 30.0  # Sensor timeout
                records_processed = 0
                retry_count = 3
                memory_mb = round(base_mem + 5.0, 2)
                scenario = "MISSING_S3_FILE"
                status = 1

        elif i in [44, 45, 46]:  # Sequence leading to Data Quality Null Failure
            if i < 46:
                duration = round(random.uniform(2.5, 3.8), 2)
                records_processed = 10
                retry_count = 0
                memory_mb = round(base_mem + random.uniform(15, 30), 2)
                scenario = "NONE"
                status = 0
            else:
                duration = round(random.uniform(1.8, 2.5), 2)
                records_processed = 0
                retry_count = 0
                memory_mb = round(base_mem + random.uniform(20, 35), 2)
                scenario = "DATA_QUALITY_NULL"
                status = 1
                
        elif i in [60, 61, 62]:  # Another periodic network glitch
            if i < 62:
                duration = round(random.uniform(4.2, 6.0), 2)
                records_processed = 10
                retry_count = 2
                memory_mb = round(base_mem + random.uniform(8, 16), 2)
                scenario = "NONE"
                status = 0
            else:
                duration = 10.0
                records_processed = 0
                retry_count = 5
                memory_mb = round(base_mem + 10.0, 2)
                scenario = "API_TIMEOUT"
                status = 1
                
        elif i == 72:  # Occasional corrupted batch
            duration = round(random.uniform(1.9, 2.4), 2)
            records_processed = 0
            retry_count = 0
            memory_mb = round(base_mem + 22.0, 2)
            scenario = "DATA_QUALITY_NULL"
            status = 1
            
        else:
            # Normal healthy pipeline run
            duration = round(random.uniform(1.6, 2.8), 2)
            records_processed = 10
            retry_count = 0
            memory_mb = round(base_mem + random.uniform(-2, 10), 2)
            scenario = "NONE"
            status = 0
            
        records.append({
            "run_id": run_id,
            "timestamp": timestamp,
            "duration": duration,
            "records_processed": records_processed,
            "retry_count": retry_count,
            "memory_mb": memory_mb,
            "failure_scenario": scenario,
            "status": status
        })
        
        current_dt += timedelta(days=1, hours=random.randint(0, 2))
        
    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Generated {len(df)} telemetry runs in {OUTPUT_CSV}")
    print(f"Success runs: {(df['status'] == 0).sum()} | Failure runs: {(df['status'] == 1).sum()}")
    return df

if __name__ == "__main__":
    generate_telemetry_dataset()

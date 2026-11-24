#!/usr/bin/env python3
"""
demo.py — End-to-End Live Demonstration Script
Data Pipeline Failure Prediction System (ANNA Course 2307411T - Track 2: Sequential Models)

Group 12 (Equal Contributors):
 - Ayush Nagras     (PRN: 202301070001)
 - Siddhesh Bhivare (PRN: 202301070005)
 - Om Nikam         (PRN: 202301070006)
 - Vaibhav Tajane   (PRN: 202301070008)

Demonstrates the Complete Multi-Stage Architecture:
1. Data Engineering ETL Pipeline Execution (Airflow Producer + Consumer Flow)
2. Telemetry Ingestion & Relational Schema Inspection (log_id Key Linkage)
3. Natural Airflow Log Analysis Engine (Root Cause Diagnosis from Stack Traces)
4. Sequential Neural Networks Benchmark (Vanilla RNN vs. LSTM vs. GRU)
5. Week 2 Advanced Temporal Attention Network (Attention-GRU Dual-Head Risk & Component Diagnosis)
"""

import os
import sys
import time
import json
import psutil
import requests
import torch
import pandas as pd
from datetime import datetime

# Path setup
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

from src.features.sequence_generator import prepare_sequential_data
from src.models.train import run_model_comparison
from src.models.train_attention import run_attention_experiment
from src.models.attention_model import AttentionGRU
from src.models.attention_visualizer import format_attention_report, generate_prediction_narrative
from src.data.log_analyzer import get_run_analysis_by_log_id

def step_1_run_pipeline():
    print("\n" + "="*80)
    print(" [STEP 1/4] DATA ENGINEERING ETL PIPELINE EXECUTION (AIRFLOW DAG FLOW)")
    print("="*80)
    
    start_time = time.time()
    process = psutil.Process(os.getpid())
    
    # 1. Producer: Fetch API data
    print(" • [Producer DAG] Fetching daily batch from JSONPlaceholder upstream REST API...")
    resp = requests.get("https://jsonplaceholder.typicode.com/users", timeout=10)
    resp.raise_for_status()
    raw_records = resp.json()
    print(f"   ✓ Extracted {len(raw_records)} records from upstream API.")
    
    # Save raw staging
    raw_path = os.path.join(PROJECT_ROOT, "data", "raw", "users_demo.json")
    with open(raw_path, "w") as f:
        json.dump(raw_records, f, indent=2)
    print(f"   ✓ Staged raw payload to: data/raw/users_demo.json")
        
    # 2. Consumer: Validation & Transformation (PySpark & Postgres logic)
    print(" • [Consumer DAG] Validating schema integrity & applying transformations...")
    df = pd.DataFrame(raw_records)
    null_count = df['id'].isna().sum() + df['email'].isna().sum()
    if null_count > 0:
        raise ValueError("Data quality check failed!")
    print(f"   ✓ Data Quality Check Passed: 0 null values in primary fields (id, email).")
    
    df_clean = df[['id', 'name', 'email']]
    processed_path = os.path.join(PROJECT_ROOT, "data", "processed", "users_demo.csv")
    df_clean.to_csv(processed_path, index=False)
    print(f"   ✓ Transformed {len(df_clean)} cleaned records saved to: data/processed/users_demo.csv")
    
    elapsed = round(time.time() - start_time, 2)
    mem_mb = round(process.memory_info().rss / (1024 * 1024), 2)
    print(f" • Pipeline Execution Completed in {elapsed}s (Memory: {mem_mb} MB).")

def step_2_show_telemetry():
    print("\n" + "="*80)
    print(" [STEP 2/4] TELEMETRY DATASET INSPECTION (data/raw/pipeline_telemetry.csv)")
    print("="*80)
    telemetry_path = os.path.join(PROJECT_ROOT, "data", "raw", "pipeline_telemetry.csv")
    df_telem = pd.read_csv(telemetry_path)
    print(f" • Total Historical Pipeline Executions Logged: {len(df_telem)} runs")
    print(f" • Normal Runs (status=0): {(df_telem['status'] == 0).sum()} | Failed Runs (status=1): {(df_telem['status'] == 1).sum()}")
    print(" • Relational Architecture: 'log_id' column serves as Foreign Key to pipeline_logs.json")
    print("\n Sample 5 Historical Pipeline Execution Telemetry Rows:")
    sample_cols = ['run_id', 'duration', 'records_processed', 'retry_count', 'memory_mb', 'failure_scenario', 'status', 'log_id']
    print(df_telem.tail(5)[sample_cols].to_string(index=False))

def step_3_demonstrate_log_analyzer():
    print("\n" + "="*80)
    print(" [STEP 3/4] LOG ANALYSIS ENGINE (PRIMARY/FOREIGN KEY JOIN ON 'log_id')")
    print("="*80)
    telemetry_path = os.path.join(PROJECT_ROOT, "data", "raw", "pipeline_telemetry.csv")
    df_telem = pd.read_csv(telemetry_path)
    
    # Pick a sample failed run
    failed_runs = df_telem[df_telem['status'] == 1]
    sample_run = failed_runs.iloc[0]
    log_id = sample_run['log_id']
    
    print(f" • Demonstrating Log Lookup for Failed Run: {sample_run['run_id']}")
    print(f"   - Telemetry Anomaly Detected: Duration={sample_run['duration']}s, Retries={sample_run['retry_count']}")
    print(f"   - Querying pipeline_logs.json using Foreign Key: '{log_id}'")
    
    analysis = get_run_analysis_by_log_id(log_id)
    raw_log = analysis["raw_log"]
    
    print("\n [Raw Airflow Execution Log Excerpt]:")
    print("-" * 70)
    print("\n".join(raw_log.splitlines()[-9:]))
    print("-" * 70)
    
    print("\n [Automated Root Cause Diagnosis]:")
    print(f"   • Pipeline Stage:       {analysis['stage']}")
    print(f"   • Error Signature:      {analysis['error_signature']}")
    print(f"   • Root Cause:           {analysis['root_cause']}")
    print(f"   • Recommended Action:   {analysis['recommendation']}")

def step_4_run_models():
    print("\n" + "="*80)
    print(" [STEP 4/5] BASELINE RECURRENT MODELS BENCHMARK (RNN vs. LSTM vs. GRU)")
    print("="*80)
    run_model_comparison()

def step_5_run_attention_xai():
    print("\n" + "="*80)
    print(" [STEP 5/5] ADVANCED TEMPORAL ATTENTION MECHANISM & MULTI-HEAD XAI (CO3)")
    print("="*80)
    run_attention_experiment()

if __name__ == "__main__":
    print("\n" + "#"*80)
    print(" DATA PIPELINE FAILURE PREDICTION SYSTEM — END-TO-END DEMO")
    print(" Course: 2307411T - Advanced Neural Network Architectures (ANNA) | Track 2")
    print(" Group 12: Ayush Nagras, Siddhesh Bhivare, Om Nikam, Vaibhav Tajane")
    print("#"*80)
    
    step_1_run_pipeline()
    step_2_show_telemetry()
    step_3_demonstrate_log_analyzer()
    step_4_run_models()
    step_5_run_attention_xai()
    
    print("\n All demonstration steps completed successfully.")
    print("#"*80 + "\n")

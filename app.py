"""
app.py — Production Streamlit Dashboard
Data Pipeline Failure Prediction System (ANNA Course 2307411T - Track 2: Sequential Models)

Group 12:
 - Siddhesh Bhivare (PRN: 202301070005)
 - Om Nikam         (PRN: 202301070006)
 - Vaibhav Tajane   (PRN: 202301070008)
 - Ayush Nagras     (PRN: 202301070001)

Features:
1. Live Pipeline Telemetry Monitor & Pre-Failure Early Warning Alert
2. Dual-Head Attention-GRU Inference (Risk Probability + Subsystem Root Cause)
3. Explainable AI (XAI) Temporal Attention Heatmap over [t-3, t-2, t-1]
4. Primary/Foreign Key Relational Log Viewer (pipeline_telemetry.csv -> pipeline_logs.json)
5. Interactive "What-If" Scenario Simulator with live degradation sliders, presets, and CSV batch upload
6. Real-time Airflow Execution Log Diagnostics Parser (extracts Exception, Root Cause, & Action)
7. Animated Airflow ETL Pipeline inspector with visual stage-by-stage simulator & failure injection
"""

import os
import sys
import json
import time
import io
import torch
import numpy as np
import pandas as pd
import streamlit as st

# Setup paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Evict any colliding or stale 'src' package from sys.modules
if 'src' in sys.modules:
    src_mod = sys.modules['src']
    if not (hasattr(src_mod, '__file__') and src_mod.__file__ and src_mod.__file__.startswith(PROJECT_ROOT)):
        del sys.modules['src']
for mod_name in list(sys.modules.keys()):
    if mod_name.startswith('src.') and not (hasattr(sys.modules[mod_name], '__file__') and sys.modules[mod_name].__file__ and sys.modules[mod_name].__file__.startswith(PROJECT_ROOT)):
        del sys.modules[mod_name]

from src.models.attention_model import AttentionGRU
from src.features.sequence_generator import FEATURE_COLUMNS, STAGE_NAMES
from src.data.log_analyzer import get_run_analysis_by_log_id, load_logs_database, parse_log_root_cause

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Data Pipeline Failure Prediction | Group 12",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Enterprise Theme (Slate Navy / Royal Blue Palette)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main Title & Subtitle */
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #0F172A;
        letter-spacing: -0.025em;
        margin-bottom: 4px;
    }
    .sub-header {
        font-size: 0.95rem;
        color: #475569;
        margin-bottom: 24px;
        font-weight: 400;
    }
    
    /* Section Headings */
    .section-title {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #475569;
        margin-top: 18px;
        margin-bottom: 12px;
        padding-bottom: 6px;
        border-bottom: 1px solid #E2E8F0;
    }
    
    /* Metric & Insight Panels */
    .metric-panel {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 14px 16px;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        margin-bottom: 10px;
    }
    .metric-panel-label {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #64748B;
        margin-bottom: 4px;
    }
    .metric-panel-value {
        font-size: 1.45rem;
        font-weight: 700;
        color: #0F172A;
    }
    
    /* Professional Status Badges */
    .badge-critical {
        background-color: #FEF2F2;
        color: #991B1B;
        border: 1px solid #FCA5A5;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.82rem;
        display: inline-block;
    }
    .badge-elevated {
        background-color: #FFFBEB;
        color: #92400E;
        border: 1px solid #FCD34D;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.82rem;
        display: inline-block;
    }
    .badge-nominal {
        background-color: #F0FDF4;
        color: #166534;
        border: 1px solid #86EFAC;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.82rem;
        display: inline-block;
    }
    .badge-stage {
        background-color: #EFF6FF;
        color: #1E40AF;
        border: 1px solid #BFDBFE;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.82rem;
        display: inline-block;
    }
    
    /* Diagnostic Detail Card */
    .diagnostic-panel {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        font-size: 0.90rem;
    }
    
    /* Plain-English Explainer Card */
    .explainer-box {
        background: #F8FAFC;
        border-left: 4px solid #2563EB;
        padding: 14px 18px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 20px;
        font-size: 0.92rem;
        line-height: 1.55;
        color: #334155;
    }
    
    /* Step Cards for Pipeline Inspector */
    .step-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 14px;
        height: 100%;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
    }
    .step-card-num {
        font-size: 0.75rem;
        font-weight: 700;
        color: #2563EB;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .step-card-title {
        font-size: 1.0rem;
        font-weight: 600;
        color: #0F172A;
        margin-bottom: 8px;
    }
    .step-card-desc {
        font-size: 0.85rem;
        color: #64748B;
        line-height: 1.45;
    }
    
    /* Terminal Box for Airflow Execution Logs */
    .log-container {
        background-color: #0B1120;
        color: #E2E8F0;
        font-family: 'JetBrains Mono', 'Courier New', monospace;
        padding: 16px;
        border: 1px solid #1E293B;
        border-radius: 8px;
        font-size: 0.80rem;
        line-height: 1.5;
        max-height: 270px;
        overflow-y: auto;
    }
    
    /* Streamlit Tab Customization */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        border-bottom: 1px solid #E2E8F0;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 500;
        font-size: 0.92rem;
        padding: 8px 16px;
        color: #64748B;
        border-radius: 6px 6px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #EFF6FF !important;
        color: #1D4ED8 !important;
        font-weight: 600 !important;
        border-bottom: 2px solid #2563EB !important;
    }

    /* Animated Pipeline Visual Canvas */
    .pipeline-canvas {
        background: linear-gradient(135deg, #0B1120 0%, #1E293B 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 24px 20px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        color: #F8FAFC;
    }
    .pipeline-nodes-wrapper {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 8px;
    }
    .stage-node-box {
        flex: 1;
        background: rgba(30, 41, 59, 0.9);
        border: 1.5px solid #475569;
        border-radius: 10px;
        padding: 14px 10px;
        text-align: center;
        transition: all 0.3s ease;
        position: relative;
    }
    .node-active-green {
        border-color: #10B981 !important;
        animation: pulse-green 2.2s infinite;
    }
    .node-active-danger {
        border-color: #EF4444 !important;
        animation: pulse-danger 1.6s infinite;
    }
    .node-active-warn {
        border-color: #F59E0B !important;
        animation: pulse-warn 1.8s infinite;
    }
    .node-skipped {
        opacity: 0.40;
        border-color: #334155 !important;
        border-style: dashed !important;
    }
    @keyframes pulse-green {
        0%, 100% { box-shadow: 0 0 8px rgba(16, 185, 129, 0.25); }
        50% { box-shadow: 0 0 20px rgba(16, 185, 129, 0.65); }
    }
    @keyframes pulse-danger {
        0%, 100% { box-shadow: 0 0 8px rgba(239, 68, 68, 0.3); }
        50% { box-shadow: 0 0 24px rgba(239, 68, 68, 0.75); }
    }
    @keyframes pulse-warn {
        0%, 100% { box-shadow: 0 0 8px rgba(245, 158, 11, 0.3); }
        50% { box-shadow: 0 0 20px rgba(245, 158, 11, 0.65); }
    }
    .pipe-connector {
        width: 32px;
        height: 6px;
        background: #334155;
        position: relative;
        border-radius: 3px;
        overflow: hidden;
    }
    .pipe-active-green {
        background: #064E3B;
    }
    .pipe-active-green::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        height: 100%;
        width: 12px;
        background: #34D399;
        box-shadow: 0 0 8px #34D399;
        border-radius: 3px;
        animation: packet-slide 1.3s infinite linear;
    }
    .pipe-blocked {
        background: #450A0A;
        border: 1px dashed #EF4444;
    }
    @keyframes packet-slide {
        0% { transform: translateX(-15px); opacity: 0; }
        30% { opacity: 1; }
        70% { opacity: 1; }
        100% { transform: translateX(35px); opacity: 0; }
    }

    /* Dynamic Remediation & Problem Diagnosis Card */
    .remediation-card {
        background-color: #FFFFFF;
        border: 1.5px solid #CBD5E1;
        border-radius: 10px;
        padding: 20px 22px;
        margin-top: 18px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .remediation-critical {
        border-color: #F87171 !important;
        border-left: 6px solid #DC2626 !important;
    }
    .remediation-elevated {
        border-color: #FBBF24 !important;
        border-left: 6px solid #D97706 !important;
    }
    .remediation-nominal {
        border-color: #86EFAC !important;
        border-left: 6px solid #16A34A !important;
    }
    .remediation-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 6px;
    }
    .remediation-section-hdr {
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #475569;
        margin-top: 12px;
        margin-bottom: 4px;
    }
</style>
""", unsafe_allow_html=True)

def extract_feature_values(window_features):
    """
    Extracts continuous duration, retry_count, and memory_mb values
    from DataFrame, list of dicts, or numpy array.
    """
    if hasattr(window_features, "to_dict"):
        records = window_features.to_dict(orient="records")
    elif isinstance(window_features, list):
        records = window_features
    elif hasattr(window_features, "shape"):
        records = [
            {"duration": float(window_features[i, 0]), "retry_count": int(window_features[i, 2]), "memory_mb": float(window_features[i, 3])}
            for i in range(len(window_features))
        ]
    else:
        records = []
    
    if len(records) >= 3:
        d1 = float(records[-3].get("duration", 0))
        d2 = float(records[-2].get("duration", 0))
        d3 = float(records[-1].get("duration", 0))
        r1 = int(records[-3].get("retry_count", 0))
        r2 = int(records[-2].get("retry_count", 0))
        r3 = int(records[-1].get("retry_count", 0))
        m1 = float(records[-3].get("memory_mb", 0))
        m2 = float(records[-2].get("memory_mb", 0))
        m3 = float(records[-1].get("memory_mb", 0))
    else:
        d1, d2, d3 = 1.8, 1.9, 1.7
        r1, r2, r3 = 0, 0, 0
        m1, m2, m3 = 84.0, 85.0, 83.5
    return (d1, d2, d3), (r1, r2, r3), (m1, m2, m3)

def generate_dynamic_remediation(risk_val, top_subsystem, conf, attn_weights, window_features):
    """
    Synthesizes a dynamic, context-aware operational diagnosis:
    1. What is the issue? (Operational Problem Signature)
    2. What is causing the problem? (Dynamic Root-Cause Analysis combining Attention & Telemetry Drift)
    3. Actionable Engineering Solution (Step-by-step remediation to restore pipeline health)
    """
    (d1, d2, d3), (r1, r2, r3), (m1, m2, m3) = extract_feature_values(window_features)
    
    attn_list = [float(w) for w in attn_weights] if hasattr(attn_weights, "__iter__") else [0.33, 0.33, 0.34]
    max_idx = int(attn_list.index(max(attn_list))) if attn_list else 2
    dom_step_name = f"t-{3 - max_idx}"
    dom_step_pct = float(attn_list[max_idx]) * 100 if attn_list else 33.3

    if risk_val >= 0.40:
        alert_tag = "CRITICAL HIGH RISK"
        alert_class = "critical"
    elif risk_val >= 0.15:
        alert_tag = "ELEVATED RISK"
        alert_class = "elevated"
    else:
        alert_tag = "NOMINAL SLA"
        alert_class = "nominal"

    # Case 1: Healthy Nominal Execution
    if risk_val < 0.15:
        return {
            "alert_tag": alert_tag,
            "alert_class": alert_class,
            "issue_title": "Nominal Operational SLA (Healthy Pipeline Execution)",
            "issue_summary": "All telemetry metrics across the 3-day observation window reside comfortably within nominal SLA bounds. No resource exhaustion, upstream socket stalls, or retry cascades detected.",
            "root_cause_explanation": (
                f"Continuous baseline stability maintained across observation window [{d1:.1f}s to {d3:.1f}s duration]. "
                f"Worker memory remains well below safe capacity ({m3:.1f} MB < 120 MB), retries are zero, and attention "
                f"distribution is uniformly balanced. The neural network assigns {conf:.1%} confidence to healthy SLA execution."
            ),
            "remediation_steps": [
                "Maintain standard @daily Airflow DAG scheduling cadence without manual intervention.",
                "Sustain current PySpark allocation (1G RAM) and API request timeout parameters (10s).",
                "Continuous automated telemetry surveillance remains active."
            ],
            "automated_action_label": "Verify Pipeline SLA Metrics",
            "automated_action_result": "Automated Check Complete: All 4 pipeline stages verified healthy. Zero intervention needed."
        }

    # Case 2: Memory Leak / PySpark Heap Pressure
    mem_drift = ((m3 - m1) / max(m1, 1.0)) * 100
    if m3 >= 135.0 or mem_drift >= 35.0 or (top_subsystem == "SPARK_PROCESSING" and m3 > 100.0):
        return {
            "alert_tag": alert_tag,
            "alert_class": alert_class,
            "issue_title": "Escalating JVM Memory Leak & Spark Executor Heap Pressure",
            "issue_summary": (
                f"Critical memory accumulation detected across consecutive executions. Worker memory surged from "
                f"{m1:.1f} MB (Run t-3) to {m3:.1f} MB (Run t-1), representing a +{mem_drift:.1f}% drift that threatens "
                f"imminent Spark OutOfMemoryError (OOM)."
            ),
            "root_cause_explanation": (
                f"The Temporal Attention mechanism concentrates {dom_step_pct:.1f}% of predictive weight on Run {dom_step_name}, "
                f"identifying it as the primary inflection point of progressive heap starvation. Uncollected DataFrame objects "
                f"or partition skew are overwhelming JVM garbage collection threads, delaying task completion and escalating failure probability to {risk_val:.1%}."
            ),
            "remediation_steps": [
                "Scale Executor Heap: Increase spark.executor.memory from 1G to 4G and spark.driver.memory to 2G in spark_submit configurations.",
                "Enable G1 Garbage Collector: Configure -XX:+UseG1GC -XX:InitiatingHeapOccupancyPercent=35 to efficiently collect fragmented heap partitions.",
                "Prevent Partition Skew: Add .repartition(8) prior to wide-transformation joins to distribute memory loads evenly across worker nodes.",
                "Unpersist Intermediate DataFrames: Audit transformation logic to ensure cached RDDs/DataFrames execute .unpersist() following write actions."
            ],
            "automated_action_label": "Execute Dynamic PySpark Heap Scaling (1G -> 4G)",
            "automated_action_result": "Automated Remediation Dispatched: Scaled spark.executor.memory to 4096MB and triggered JVM GC sweep. Next run projected to pass SLA."
        }

    # Case 3: Upstream REST API Ingestion Latency / Timeout Storm
    if r3 >= 1 or d3 >= 6.0 or top_subsystem == "API_INGESTION":
        dur_drift = ((d3 - d1) / max(d1, 0.1)) * 100
        return {
            "alert_tag": alert_tag,
            "alert_class": alert_class,
            "issue_title": "Upstream REST API Connection Latency & Retry Queue Exhaustion",
            "issue_summary": (
                f"Severe upstream network friction detected in the ingestion stage. Execution duration escalated from "
                f"{d1:.2f}s to {d3:.2f}s (+{dur_drift:.1f}%), triggering {r3} task retry attempts on Run t-1."
            ),
            "root_cause_explanation": (
                f"Run {dom_step_name} accounts for {dom_step_pct:.1f}% of total temporal attention. External endpoint "
                f"(JSONPlaceholder API) is exhibiting high response latency exceeding the default socket read threshold (10s), "
                f"initiating exponential backoff retry cycles that exhaust Airflow worker slots and push failure risk to {risk_val:.1%}."
            ),
            "remediation_steps": [
                "Extend Socket Timeout: Increase HTTP read/connect timeout in producer_users_daily.py from 10s to 30s.",
                "Implement Jittered Exponential Backoff: Configure urllib3.util.Retry(total=5, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504]).",
                "Deploy Ingestion Edge Cache: Configure Redis caching layer to serve warm cached payloads during external server degradation.",
                "Audit Provider Rate Limits: Check whether ingestion IP has approached daily rate-limiting quotas (HTTP 429 Too Many Requests)."
            ],
            "automated_action_label": "Apply HTTP Circuit Breaker & Extend Read Timeout (10s -> 30s)",
            "automated_action_result": "Automated Remediation Dispatched: Extended socket timeout to 30s and enabled circuit breaker fallback cache. Ingestion SLA secured."
        }

    # Case 4: PostgreSQL Database Lock / Warehouse Bottleneck
    if top_subsystem == "DATABASE_LOAD":
        return {
            "alert_tag": alert_tag,
            "alert_class": alert_class,
            "issue_title": "PostgreSQL Connection Pool Exhaustion & Row-Lock Contention",
            "issue_summary": (
                f"Downstream data warehouse write bottleneck detected. Target PostgreSQL database is experiencing connection saturation "
                f"or transactional row-lock queuing at Port 5432."
            ),
            "root_cause_explanation": (
                f"Model identifies {top_subsystem} as the critical root cause ({conf:.1%} confidence), with {dom_step_pct:.1f}% "
                f"attention focus centered on Run {dom_step_name}. High concurrency or uncommitted locks during idempotent upserts are blocking worker thread commits."
            ),
            "remediation_steps": [
                "Deploy Connection Pooler: Route database connections through PgBouncer with a pool size of 50 to eliminate backend thread exhaustion.",
                "Tune Transaction Parameters: Increase max_connections to 200 and set lock_timeout='15s' in postgresql.conf.",
                "Optimize Idempotent Indexing: Ensure composite index exists on (id, extraction_date) to allow row-level upsert without table locks.",
                "Purge Idle Connections: Terminate orphaned connections in pg_stat_activity holding active transactions for >60 seconds."
            ],
            "automated_action_label": "Recycle Database Connection Pool & Flush Idle Locks",
            "automated_action_result": "Automated Remediation Dispatched: Cleared PostgreSQL idle-in-transaction locks and recycled connection pool."
        }

    # Case 5: Default General Elevated / Critical Degradation
    return {
        "alert_tag": alert_tag,
        "alert_class": alert_class,
        "issue_title": f"Compound Operational Anomaly Detected ({top_subsystem})",
        "issue_summary": (
            f"Cross-stage operational friction detected across the telemetry trajectory. Neural network flags {top_subsystem} "
            f"as the primary degradation driver with an elevated risk score of {risk_val:.1%}."
        ),
        "root_cause_explanation": (
            f"The temporal attention layer allocates {dom_step_pct:.1f}% of predictive focus to Run {dom_step_name}. Telemetry signals "
            f"indicate progressive deviation from baseline operating distributions, requiring preventive intervention before complete SLA breach."
        ),
        "remediation_steps": [
            f"Inspect {top_subsystem} Stage Logs: Review raw Airflow worker logs for stack trace exceptions or timeout warnings.",
            "Verify Pipeline Dependencies: Ensure upstream network availability, S3 bucket storage quotas, and PostgreSQL connectivity.",
            "Engage Preventive Worker Scale-Out: Provision additional worker capacity to absorb load surges."
        ],
        "automated_action_label": "Dispatch Automated Pipeline Health Audit",
        "automated_action_result": "Automated Check Dispatched: Health audit initiated across all 4 stages. Diagnostics logged to Airflow control plane."
    }

def render_dynamic_remediation_card(remed_info, key_suffix="default"):
    """
    Renders a dynamic, professional diagnostic card detailing:
    - What is the issue? (Operational Problem Signature)
    - What is causing the problem? (Dynamic Root-Cause Analysis combining Attention & Telemetry Drift)
    - Prescribed Engineering Solution (Numbered Actionable Steps)
    - Interactive Self-Healing Simulation Action
    """
    alert_cls = remed_info.get("alert_class", "nominal")
    alert_tag = remed_info.get("alert_tag", "NOMINAL SLA")
    title = remed_info.get("issue_title", "")
    summary = remed_info.get("issue_summary", "")
    root_cause = remed_info.get("root_cause_explanation", "")
    steps = remed_info.get("remediation_steps", [])
    btn_label = remed_info.get("automated_action_label", "Apply Remediation")
    btn_res = remed_info.get("automated_action_result", "Action applied successfully.")
    
    steps_html = "".join([
        f"<li style='margin-bottom: 5px;'><strong>{s.split(':', 1)[0]}:</strong>{s.split(':', 1)[1]}</li>"
        if ":" in s else f"<li style='margin-bottom: 5px;'>{s}</li>"
        for s in steps
    ])
    
    st.markdown(f"""
    <div class='remediation-card remediation-{alert_cls}'>
        <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;'>
            <span class='badge-{alert_cls}'>{alert_tag}</span>
            <span style='font-size: 0.78rem; font-weight: 600; color: #64748B; letter-spacing: 0.05em;'>DYNAMIC XAI DIAGNOSTIC ENGINE</span>
        </div>
        <div class='remediation-title'>{title}</div>
        <div style='font-size: 0.90rem; color: #334155; line-height: 1.5; margin-bottom: 12px;'>
            {summary}
        </div>
        <div class='remediation-section-hdr'>What is Causing the Problem (Root Cause Analysis):</div>
        <div style='font-size: 0.88rem; color: #1E293B; line-height: 1.55; background: #F8FAFC; padding: 12px 14px; border-radius: 6px; border: 1px solid #E2E8F0; margin-bottom: 12px;'>
            {root_cause}
        </div>
        <div class='remediation-section-hdr'>Prescribed Engineering Solution & Mitigation Steps:</div>
        <ol style='font-size: 0.88rem; color: #334155; line-height: 1.55; margin-left: -15px; margin-bottom: 8px;'>
            {steps_html}
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    col_act1, col_act2 = st.columns([1.8, 2.2])
    with col_act1:
        if st.button(f"Simulate Action: {btn_label}", key=f"btn_remed_{key_suffix}"):
            st.success(f"{btn_res}")

# -----------------------------------------------------------------------------
# Data & Model Loaders
# -----------------------------------------------------------------------------
@st.cache_data
def load_historical_telemetry():
    csv_path = os.path.join(PROJECT_ROOT, "data", "raw", "pipeline_telemetry.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Telemetry CSV not found at {csv_path}. Run generate_dataset.py first.")
    df = pd.read_csv(csv_path)
    return df

@st.cache_data
def load_all_logs():
    return load_logs_database()

@st.cache_resource
def load_trained_system():
    ckpt_path = os.path.join(PROJECT_ROOT, "models", "best_attention_model.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Model checkpoint not found at {ckpt_path}. Run train_attention.py first.")
        
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    scaler = ckpt["scaler"]
    
    model = AttentionGRU(
        input_dim=ckpt.get("input_dim", 4),
        hidden_dim=ckpt.get("hidden_dim", 24),
        attention_dim=ckpt.get("attention_dim", 16),
        num_classes=ckpt.get("num_classes", 5)
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    return model, scaler

# Load assets
df_telem = load_historical_telemetry()
logs_dict = load_all_logs()
model, scaler = load_trained_system()

# -----------------------------------------------------------------------------
# Sidebar: Metadata & Academic Project Credentials
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### System Context")
    st.markdown("**Course:** 2307411T - ANNA")
    st.markdown("**Track:** Track 2 (Sequential Models)")
    st.markdown("**Assessment:** End-Semester Exam (50M)")
    st.divider()
    
    st.subheader("Group 12")
    st.markdown("• **Siddhesh Bhivare** (202301070005)")
    st.markdown("• **Om Nikam** (202301070006)")
    st.markdown("• **Vaibhav Tajane** (202301070008)")
    st.markdown("• **Ayush Nagras** (202301070001)")
    st.divider()
    
    st.subheader("System Status")
    st.success("Attention-GRU: LOADED")
    st.info(f"Historical Runs: {len(df_telem)}")
    st.info(f"Airflow Task Logs: {len(logs_dict)}")

# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.markdown("<div class='main-header'>Data Pipeline Failure Prediction System</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Pre-Failure Early Warning & Root Cause Attribution using Dual-Head Temporal Attention GRUs</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Tabs (3 Active Sections)
# -----------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "Live Risk Monitor",
    "What-If Simulator",
    "Airflow Pipeline Inspector"
])

# =============================================================================
# TAB 1: Live Risk Monitor
# =============================================================================
with tab1:
    st.subheader("Historical Execution Telemetry & Sequential Risk Assessment")
    
    # Select Run Selector
    failed_runs = df_telem[df_telem['status'] == 1]['run_id'].tolist()
    healthy_runs = df_telem[df_telem['status'] == 0]['run_id'].tail(15).tolist()
    sample_options = failed_runs[:8] + healthy_runs[:7]
    
    selected_run = st.selectbox(
        "Select Pipeline Run to Inspect Pre-Failure Trajectory:",
        options=sample_options,
        index=0
    )
    
    # Locate index
    run_idx = df_telem[df_telem['run_id'] == selected_run].index[0]
    
    if run_idx < 3:
        st.warning("Selected run is within the initial warm-up period (T < 3). Showing first valid sliding window [t-3, t-2, t-1].")
        window_df = df_telem.iloc[0:3]
        target_row = df_telem.iloc[3]
    else:
        window_df = df_telem.iloc[run_idx-3:run_idx]
        target_row = df_telem.iloc[run_idx]
        
    st.markdown("<div class='section-title'>1. Historical Telemetry Sequence Input [t-3, t-2, t-1]</div>", unsafe_allow_html=True)
    cols_display = ['run_id', 'duration', 'records_processed', 'retry_count', 'memory_mb', 'failure_scenario', 'log_id']
    st.dataframe(window_df[cols_display], hide_index=True)
    
    # Prepare sequence for inference with feature names to prevent sklearn warning
    feat_values = scaler.transform(window_df[FEATURE_COLUMNS])
    seq_tensor = torch.tensor(feat_values, dtype=torch.float32).unsqueeze(0)  # (1, 3, 4)
    
    # Inference
    with torch.no_grad():
        risk_prob, cause_logits, attn_weights = model(seq_tensor)
        risk_val = float(risk_prob.item())
        cause_probs = torch.softmax(cause_logits, dim=1).squeeze(0).numpy()
        attn_np = attn_weights.squeeze(0).numpy()
        
    top_stage_idx = int(np.argmax(cause_probs))
    top_stage_name = STAGE_NAMES[top_stage_idx]
    top_stage_conf = float(cause_probs[top_stage_idx])
    
    st.markdown("<div class='section-title'>2. Dual-Head Model Inference & Operational Early Warning</div>", unsafe_allow_html=True)
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    
    with col_m1:
        st.markdown(f"""
        <div class='metric-panel'>
            <div class='metric-panel-label'>Predicted Risk Probability</div>
            <div class='metric-panel-value'>{risk_val:.1%}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_m2:
        if risk_val >= 0.40:
            badge_html = "<span class='badge-critical'>CRITICAL HIGH RISK</span>"
        elif risk_val >= 0.15:
            badge_html = "<span class='badge-elevated'>ELEVATED RISK</span>"
        else:
            badge_html = "<span class='badge-nominal'>NOMINAL SLA</span>"
            
        st.markdown(f"""
        <div class='metric-panel'>
            <div class='metric-panel-label'>System Alert Level</div>
            <div style='margin-top: 6px;'>{badge_html}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_m3:
        st.markdown(f"""
        <div class='metric-panel'>
            <div class='metric-panel-label'>Attributed Subsystem</div>
            <div style='margin-top: 6px;'><span class='badge-stage'>{top_stage_name}</span></div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_m4:
        st.markdown(f"""
        <div class='metric-panel'>
            <div class='metric-panel-label'>Subsystem Confidence</div>
            <div class='metric-panel-value'>{top_stage_conf:.1%}</div>
        </div>
        """, unsafe_allow_html=True)
        
    # Temporal Attention Breakdown
    st.markdown("<div class='section-title'>3. Temporal Attention Weights (Explainable AI Attribution)</div>", unsafe_allow_html=True)
    col_attn1, col_attn2 = st.columns([1.5, 1])
    
    with col_attn1:
        attn_df = pd.DataFrame({
            "Historical Execution": [f"Run t-3 ({window_df.iloc[0]['run_id']})",
                                     f"Run t-2 ({window_df.iloc[1]['run_id']})",
                                     f"Run t-1 ({window_df.iloc[2]['run_id']})"],
            "Attention Contribution (%)": [w * 100 for w in attn_np]
        })
        st.bar_chart(attn_df, x="Historical Execution", y="Attention Contribution (%)")
        
    with col_attn2:
        max_t = int(np.argmax(attn_np))
        st.markdown(f"""
        <div class='diagnostic-panel'>
            <strong>Temporal Explainability Summary:</strong><br><br>
            • <strong>Dominant Root-Cause Driver:</strong> <code>Run t-{3-max_t}</code> accounts for <strong>{attn_np[max_t]:.1%}</strong> of total attention focus.<br>
            • <strong>Trajectory:</strong> Pre-failure telemetry anomalies initiated during this execution step.<br>
            • <strong>Target Ground Truth:</strong> <code>{target_row['failure_scenario']}</code> (Actual status: {target_row['status']})
        </div>
        """, unsafe_allow_html=True)
        
    # Dynamic Diagnostic & Actionable Remediation Panel
    st.markdown("<div class='section-title'>4. Operational Root Cause & Actionable Engineering Remediation</div>", unsafe_allow_html=True)
    remed_tab1 = generate_dynamic_remediation(
        risk_val=risk_val,
        top_subsystem=top_stage_name,
        conf=top_stage_conf,
        attn_weights=attn_np,
        window_features=window_df
    )
    render_dynamic_remediation_card(remed_tab1, key_suffix=f"tab1_{selected_run}")
        
    # Primary/Foreign Key Log Viewer
    st.markdown("<div class='section-title'>5. Relational Log Analysis (Joined on log_id)</div>", unsafe_allow_html=True)
    log_id = target_row['log_id']
    analysis = get_run_analysis_by_log_id(log_id)
    
    col_log1, col_log2 = st.columns([1, 1.2])
    with col_log1:
        st.markdown(f"""
        <div class='diagnostic-panel'>
            <strong>Database Key:</strong> <code>{log_id}</code><br><br>
            <strong>Failing Stage:</strong> <code>{analysis['stage']}</code><br>
            <strong>Exception Class:</strong> <code>{analysis['error_signature']}</code><br><br>
            <strong>Plain-English Diagnosis:</strong><br>
            {analysis['root_cause']}<br><br>
            <strong>Engineering Remediation:</strong><br>
            {analysis['recommendation']}
        </div>
        """, unsafe_allow_html=True)
        
    with col_log2:
        st.caption("Raw Multi-Task Airflow Execution Trace (from data/raw/pipeline_logs.json):")
        raw_log = analysis['raw_log']
        st.markdown(f"<div class='log-container'>{raw_log.replace('<', '&lt;').replace('>', '&gt;')}</div>", unsafe_allow_html=True)

# =============================================================================
# TAB 2: Interactive What-If Simulator & Batch CSV Evaluation
# =============================================================================
with tab2:
    st.subheader("Interactive What-If Degradation Simulator & Diagnostics")
    st.markdown("Test the neural network with manual parameter sliders, evaluate batch telemetry CSV files, or diagnose raw Airflow execution logs.")
    
    sim_mode = st.radio(
        "Select Tool Mode:",
        options=["Interactive Telemetry Sliders", "Telemetry CSV Evaluation (Batch)", "Airflow Log Text Diagnostics"],
        horizontal=True
    )
    
    if sim_mode == "Interactive Telemetry Sliders":
        # Preset Selector in vertical list view
        preset = st.radio(
            "Load Degradation Pattern Preset (Select a Scenario to Auto-Fill Inputs):",
            options=[
                "Custom Telemetry",
                "Gradual Memory Leak (Spark OOM)",
                "Upstream API Timeout Cascade",
                "Nominal Healthy Execution"
            ],
            index=0
        )
        
        # Preset dictionary mapping
        preset_values = {
            "Gradual Memory Leak (Spark OOM)": {
                "d1": 2.5, "r1": 0, "m1": 95.0,
                "d2": 4.8, "r2": 1, "m2": 130.0,
                "d3": 14.5, "r3": 2, "m3": 185.0
            },
            "Upstream API Timeout Cascade": {
                "d1": 1.63, "r1": 0, "m1": 84.0,
                "d2": 3.38, "r2": 1, "m2": 101.4,
                "d3": 7.57, "r3": 2, "m3": 114.7
            },
            "Nominal Healthy Execution": {
                "d1": 1.8, "r1": 0, "m1": 84.0,
                "d2": 1.9, "r2": 0, "m2": 85.0,
                "d3": 1.7, "r3": 0, "m3": 83.5
            },
            "Custom Telemetry": {
                "d1": 2.0, "r1": 0, "m1": 85.0,
                "d2": 3.5, "r2": 1, "m2": 95.0,
                "d3": 7.0, "r3": 2, "m3": 115.0
            }
        }
        
        # Check if preset changed, and update slider session state values so sliders move visibly
        if "active_preset" not in st.session_state:
            st.session_state["active_preset"] = preset
            
        if st.session_state["active_preset"] != preset:
            vals = preset_values.get(preset, preset_values["Custom Telemetry"])
            st.session_state["d1"] = float(vals["d1"])
            st.session_state["r1"] = int(vals["r1"])
            st.session_state["m1"] = float(vals["m1"])
            st.session_state["d2"] = float(vals["d2"])
            st.session_state["r2"] = int(vals["r2"])
            st.session_state["m2"] = float(vals["m2"])
            st.session_state["d3"] = float(vals["d3"])
            st.session_state["r3"] = int(vals["r3"])
            st.session_state["m3"] = float(vals["m3"])
            st.session_state["active_preset"] = preset

        # Ensure session state has defaults if first run
        cur_vals = preset_values.get(st.session_state["active_preset"], preset_values["Custom Telemetry"])
        if "d1" not in st.session_state: st.session_state["d1"] = float(cur_vals["d1"])
        if "r1" not in st.session_state: st.session_state["r1"] = int(cur_vals["r1"])
        if "m1" not in st.session_state: st.session_state["m1"] = float(cur_vals["m1"])
        if "d2" not in st.session_state: st.session_state["d2"] = float(cur_vals["d2"])
        if "r2" not in st.session_state: st.session_state["r2"] = int(cur_vals["r2"])
        if "m2" not in st.session_state: st.session_state["m2"] = float(cur_vals["m2"])
        if "d3" not in st.session_state: st.session_state["d3"] = float(cur_vals["d3"])
        if "r3" not in st.session_state: st.session_state["r3"] = int(cur_vals["r3"])
        if "m3" not in st.session_state: st.session_state["m3"] = float(cur_vals["m3"])
            
        col_sim1, col_sim2, col_sim3 = st.columns(3)
        
        with col_sim1:
            st.markdown("#### Run t-3 (2 Days Ago)")
            dur_1 = st.slider("Duration (s)", 0.5, 30.0, key="d1")
            ret_1 = st.slider("Retry Count", 0, 5, key="r1")
            mem_1 = st.slider("Worker Memory (MB)", 50.0, 250.0, key="m1")
            rec_1 = 10
            
        with col_sim2:
            st.markdown("#### Run t-2 (Yesterday)")
            dur_2 = st.slider("Duration (s)", 0.5, 30.0, key="d2")
            ret_2 = st.slider("Retry Count", 0, 5, key="r2")
            mem_2 = st.slider("Worker Memory (MB)", 50.0, 250.0, key="m2")
            rec_2 = 10
            
        with col_sim3:
            st.markdown("#### Run t-1 (Most Recent)")
            dur_3 = st.slider("Duration (s)", 0.5, 30.0, key="d3")
            ret_3 = st.slider("Retry Count", 0, 5, key="r3")
            mem_3 = st.slider("Worker Memory (MB)", 50.0, 250.0, key="m3")
            rec_3 = 10
            
        # Transform simulation input with feature names
        sim_df = pd.DataFrame([
            {'duration': dur_1, 'records_processed': rec_1, 'retry_count': ret_1, 'memory_mb': mem_1},
            {'duration': dur_2, 'records_processed': rec_2, 'retry_count': ret_2, 'memory_mb': mem_2},
            {'duration': dur_3, 'records_processed': rec_3, 'retry_count': ret_3, 'memory_mb': mem_3}
        ], columns=FEATURE_COLUMNS)
        sim_scaled = scaler.transform(sim_df)
        sim_tensor = torch.tensor(sim_scaled, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            sim_risk, sim_cause, sim_attn = model(sim_tensor)
            s_risk = float(sim_risk.item())
            s_cause = torch.softmax(sim_cause, dim=1).squeeze(0).numpy()
            s_attn = sim_attn.squeeze(0).numpy()
            
        st.divider()
        st.markdown("<div class='section-title'>Simulated Real-Time Prediction Results</div>", unsafe_allow_html=True)
        
        col_sr1, col_sr2, col_sr3 = st.columns([1, 1, 2])
        with col_sr1:
            if s_risk >= 0.40:
                badge_sim = "<span class='badge-critical'>CRITICAL HIGH RISK</span>"
            elif s_risk >= 0.15:
                badge_sim = "<span class='badge-elevated'>ELEVATED RISK</span>"
            else:
                badge_sim = "<span class='badge-nominal'>NOMINAL SLA</span>"
                
            st.markdown(f"""
            <div class='metric-panel'>
                <div class='metric-panel-label'>Predicted Risk Probability</div>
                <div class='metric-panel-value'>{s_risk:.1%}</div>
                <div style='margin-top: 6px;'>{badge_sim}</div>
            </div>
            """, unsafe_allow_html=True)
                
        with col_sr2:
            sim_top_stage = STAGE_NAMES[int(np.argmax(s_cause))]
            st.markdown(f"""
            <div class='metric-panel'>
                <div class='metric-panel-label'>Attributed Subsystem</div>
                <div style='margin-top: 6px;'><span class='badge-stage'>{sim_top_stage}</span></div>
                <div style='margin-top: 8px; font-size: 0.85rem; color: #64748B;'>Confidence: <strong>{np.max(s_cause):.1%}</strong></div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_sr3:
            st.caption("Temporal Attention Focus over Input Window:")
            s_df = pd.DataFrame({
                "Step": ["Run t-3", "Run t-2", "Run t-1"],
                "Weight (%)": [w * 100 for w in s_attn]
            })
            st.bar_chart(s_df, x="Step", y="Weight (%)")
            
        # Dynamic Diagnostic & Actionable Engineering Remediation Panel
        st.markdown("<div class='section-title'>Operational Root Cause & Prescribed Engineering Remediation</div>", unsafe_allow_html=True)
        remed_sim = generate_dynamic_remediation(
            risk_val=s_risk,
            top_subsystem=sim_top_stage,
            conf=float(np.max(s_cause)),
            attn_weights=s_attn,
            window_features=sim_df
        )
        render_dynamic_remediation_card(remed_sim, key_suffix="sim_sliders")

    elif sim_mode == "Telemetry CSV Evaluation (Batch)":
        st.markdown("<div class='section-title'>Batch Telemetry CSV Evaluation</div>", unsafe_allow_html=True)
        st.markdown("Upload any CSV with pipeline telemetry or select from pre-calibrated sample files to evaluate multi-run failure risk trajectories.")
        
        # Download Sample CSVs
        samples_dir = os.path.join(PROJECT_ROOT, "data", "samples")
        col_dl1, col_dl2, col_dl3 = st.columns(3)
        
        def get_sample_csv_bytes(filename):
            p = os.path.join(samples_dir, filename)
            if os.path.exists(p):
                with open(p, "rb") as f:
                    return f.read()
            return b""
            
        with col_dl1:
            st.download_button(
                label="Download High Risk CSV",
                data=get_sample_csv_bytes("sample_high_risk_failure.csv"),
                file_name="sample_high_risk_failure.csv",
                mime="text/csv"
            )
        with col_dl2:
            st.download_button(
                label="Download Medium Risk CSV",
                data=get_sample_csv_bytes("sample_medium_risk_elevated.csv"),
                file_name="sample_medium_risk_elevated.csv",
                mime="text/csv"
            )
        with col_dl3:
            st.download_button(
                label="Download Healthy Baseline CSV",
                data=get_sample_csv_bytes("sample_healthy_nominal.csv"),
                file_name="sample_healthy_nominal.csv",
                mime="text/csv"
            )
            
        st.markdown("<br>", unsafe_allow_html=True)
        csv_source = st.selectbox(
            "Select CSV Telemetry Input:",
            options=[
                "Upload Custom CSV File",
                "Built-in: High Failure Risk Sequence (Severe Escalating Degradation)",
                "Built-in: Medium Elevated Risk Sequence (Creeping Friction Warning)",
                "Built-in: Healthy Baseline Sequence (Nominal SLA Execution)"
            ]
        )
        
        user_df = None
        if csv_source == "Upload Custom CSV File":
            uploaded_file = st.file_uploader("Upload CSV containing columns (duration, records_processed, retry_count, memory_mb):", type=["csv"])
            if uploaded_file is not None:
                user_df = pd.read_csv(uploaded_file)
        elif "High Failure Risk" in csv_source:
            p = os.path.join(samples_dir, "sample_high_risk_failure.csv")
            if os.path.exists(p): user_df = pd.read_csv(p)
        elif "Medium Elevated Risk" in csv_source:
            p = os.path.join(samples_dir, "sample_medium_risk_elevated.csv")
            if os.path.exists(p): user_df = pd.read_csv(p)
        else:
            p = os.path.join(samples_dir, "sample_healthy_nominal.csv")
            if os.path.exists(p): user_df = pd.read_csv(p)
            
        if user_df is not None:
            st.markdown("##### Telemetry Data Sequence Preview")
            st.dataframe(user_df.head(10))
            
            missing_cols = [c for c in FEATURE_COLUMNS if c not in user_df.columns]
            if missing_cols:
                st.error(f"Missing required numeric columns: {missing_cols}")
            elif len(user_df) < 3:
                st.error("CSV must contain at least 3 historical runs to form a temporal sliding window.")
            else:
                st.success(f"Loaded {len(user_df)} historical pipeline executions. Evaluating with Attention-GRU...")
                
                feats_scaled = scaler.transform(user_df[FEATURE_COLUMNS])
                results_list = []
                
                for i in range(3, len(user_df) + 1):
                    window = feats_scaled[i-3:i]
                    window_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
                    
                    with torch.no_grad():
                        r_prob, c_logits, a_w = model(window_tensor)
                        risk_val = float(r_prob.item())
                        cause_probs = torch.softmax(c_logits, dim=1).squeeze(0).numpy()
                        attn_w = a_w.squeeze(0).numpy()
                        
                    target_label = user_df.iloc[i-1].get('run_id', f'Run_{i}')
                    top_subsystem = STAGE_NAMES[int(np.argmax(cause_probs))]
                    
                    risk_tag = "CRITICAL" if risk_val >= 0.40 else ("ELEVATED" if risk_val >= 0.15 else "NOMINAL")
                    
                    results_list.append({
                        "Target Run": target_label,
                        "Failure Risk": f"{risk_val:.1%}",
                        "Alert Level": risk_tag,
                        "Attributed Subsystem": top_subsystem,
                        "Attention Dominant Step": f"t-{3 - int(np.argmax(attn_w))} ({attn_w[int(np.argmax(attn_w))]:.1%})",
                        "Raw_Risk": risk_val,
                        "Cause_Probs": cause_probs,
                        "Attn_W": attn_w
                    })
                    
                res_df = pd.DataFrame(results_list)
                
                st.markdown("##### Multi-Window Sequential Model Evaluation Results")
                st.dataframe(res_df.drop(columns=['Raw_Risk', 'Cause_Probs', 'Attn_W']))
                
                # Render Simulated Real-Time Prediction Results (matching Interactive Telemetry Sliders)
                st.divider()
                st.markdown("<div class='section-title'>Simulated Real-Time Prediction Results</div>", unsafe_allow_html=True)
                
                window_options = [r["Target Run"] for r in results_list]
                selected_target = st.selectbox(
                    "Select Pipeline Run from CSV to Inspect Prediction Metrics:",
                    options=window_options,
                    index=len(window_options) - 1
                )
                
                sel_idx = window_options.index(selected_target)
                sel_res = results_list[sel_idx]
                s_risk = sel_res["Raw_Risk"]
                s_cause = sel_res["Cause_Probs"]
                s_attn = sel_res["Attn_W"]
                sim_top_stage = sel_res["Attributed Subsystem"]
                sim_top_conf = float(np.max(s_cause))
                
                col_sr1, col_sr2, col_sr3 = st.columns([1, 1, 2])
                with col_sr1:
                    if s_risk >= 0.40:
                        badge_sim = "<span class='badge-critical'>CRITICAL HIGH RISK</span>"
                    elif s_risk >= 0.15:
                        badge_sim = "<span class='badge-elevated'>ELEVATED RISK</span>"
                    else:
                        badge_sim = "<span class='badge-nominal'>NOMINAL SLA</span>"
                        
                    st.markdown(f"""
                    <div class='metric-panel'>
                        <div class='metric-panel-label'>Predicted Risk Probability</div>
                        <div class='metric-panel-value'>{s_risk:.1%}</div>
                        <div style='margin-top: 6px;'>{badge_sim}</div>
                    </div>
                    """, unsafe_allow_html=True)
                        
                with col_sr2:
                    st.markdown(f"""
                    <div class='metric-panel'>
                        <div class='metric-panel-label'>Attributed Subsystem</div>
                        <div style='margin-top: 6px;'><span class='badge-stage'>{sim_top_stage}</span></div>
                        <div style='margin-top: 8px; font-size: 0.85rem; color: #64748B;'>Confidence: <strong>{sim_top_conf:.1%}</strong></div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col_sr3:
                    st.caption("Temporal Attention Focus over Input Window:")
                    s_df = pd.DataFrame({
                        "Step": ["Run t-3", "Run t-2", "Run t-1"],
                        "Weight (%)": [w * 100 for w in s_attn]
                    })
                    st.bar_chart(s_df, x="Step", y="Weight (%)")
                    
                # Dynamic Diagnostic & Actionable Remediation Panel for Batch CSV Window
                st.markdown("<div class='section-title'>Operational Root Cause & Prescribed Engineering Remediation</div>", unsafe_allow_html=True)
                sel_row_match = user_df[user_df['run_id'] == selected_target] if 'run_id' in user_df.columns else None
                if sel_row_match is not None and not sel_row_match.empty:
                    sel_pos = sel_row_match.index[0]
                    csv_slice = user_df.iloc[max(0, sel_pos-2):sel_pos+1]
                else:
                    csv_slice = user_df.iloc[max(0, sel_idx-2):sel_idx+1]
                    
                remed_csv = generate_dynamic_remediation(
                    risk_val=s_risk,
                    top_subsystem=sim_top_stage,
                    conf=sim_top_conf,
                    attn_weights=s_attn,
                    window_features=csv_slice
                )
                render_dynamic_remediation_card(remed_csv, key_suffix=f"csv_{selected_target}")
                    
                # Relational Log Inspection for the Selected Run (if log_id or log_text column is in CSV)
                csv_log_text = None
                log_ref = None
                matched_row = user_df[user_df['run_id'] == selected_target] if 'run_id' in user_df.columns else None
                if matched_row is not None and not matched_row.empty:
                    m_row = matched_row.iloc[0]
                    if 'log_id' in m_row and pd.notna(m_row['log_id']) and str(m_row['log_id']) in logs_dict:
                        csv_log_text = logs_dict[str(m_row['log_id'])]
                        log_ref = str(m_row['log_id'])
                    elif 'raw_log' in m_row and pd.notna(m_row['raw_log']):
                        csv_log_text = str(m_row['raw_log'])
                        log_ref = "Embedded in CSV (raw_log column)"
                    elif 'log_text' in m_row and pd.notna(m_row['log_text']):
                        csv_log_text = str(m_row['log_text'])
                        log_ref = "Embedded in CSV (log_text column)"
                        
                if csv_log_text:
                    st.markdown("<div class='section-title'>Associated Task Execution Log (Joined from CSV)</div>", unsafe_allow_html=True)
                    log_diag = parse_log_root_cause(csv_log_text)
                    col_cl1, col_cl2 = st.columns([1, 1.2])
                    with col_cl1:
                        st.markdown(f"""
                        <div class='diagnostic-panel'>
                            <strong>Database Key / Reference:</strong> <code>{log_ref}</code><br><br>
                            <strong>Failing Stage:</strong> <code>{log_diag['stage']}</code><br>
                            <strong>Exception Class:</strong> <code>{log_diag['error_signature'] or 'None (Clean Run)'}</code><br><br>
                            <strong>Plain-English Root Cause:</strong><br>{log_diag['root_cause']}<br><br>
                            <strong>Actionable Engineering Remediation:</strong><br>{log_diag['recommendation']}
                        </div>
                        """, unsafe_allow_html=True)
                    with col_cl2:
                        st.caption("Active Airflow Task Execution Trace:")
                        st.markdown(f"<div class='log-container'>{csv_log_text.replace('<', '&lt;').replace('>', '&gt;')}</div>", unsafe_allow_html=True)

    else:
        # Airflow Log Text Diagnostics
        st.markdown("<div class='section-title'>Airflow Execution Log Diagnostic Parser</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='explainer-box'>
            <strong>How do Logs differ from Telemetry? (Viva Conceptual Alignment):</strong><br>
            • <strong>Telemetry (Quantitative CSV):</strong> Continuous numeric time-series (duration, memory, retries). Fed into the <strong>Attention-GRU</strong> to forecast failure risk <em>before</em> the pipeline runs.<br>
            • <strong>Task Logs (Qualitative Text):</strong> Rich execution traces and stack traces generated by Airflow workers during execution. Parsed to identify the exact <strong>Exception Signature</strong>, <strong>Root Cause</strong>, and <strong>Engineering Remediation</strong>.<br>
            • <strong>Model Connection:</strong> The Attention-GRU multi-class subsystem head was trained using ground-truth labels derived directly by parsing 450 historical task logs.
        </div>
        """, unsafe_allow_html=True)
        
        log_entry_mode = st.radio(
            "Select How You Want to Input Airflow Logs:",
            options=["Select from Realistic Error Presets", "Paste Custom Raw Log Text", "Lookup by Historical log_id"],
            horizontal=True
        )
        
        # Preset Log Traces
        preset_logs = {
            "Spark Executor JVM Out-Of-Memory (Heap Space)": (
                "[2026-11-20 04:02:11] {taskinstance.py:1150} INFO - Executing <Task(SparkSubmitOperator): transform_user_lake>\n"
                "[2026-11-20 04:02:19] {spark_submit.py:112} INFO - Spark Executor assigned 1024MB RAM.\n"
                "[2026-11-20 04:02:28] {spark_submit.py:180} ERROR - java.lang.OutOfMemoryError: Java heap space\n"
                "  at org.apache.spark.memory.UnifiedMemoryManager.acquireExecutionMemory(UnifiedMemoryManager.scala:132)\n"
                "  at org.apache.spark.sql.execution.aggregate.HashAggregateExec.doExecute(HashAggregateExec.scala:114)\n"
                "[2026-11-20 04:02:29] {taskinstance.py:1463} ERROR - Task failed with exception: Spark execution failed."
            ),
            "Upstream REST API HTTP 500 Internal Server Error": (
                "[2026-11-21 04:00:02] {taskinstance.py:1150} INFO - Executing <Task(HttpOperator): extract_users_api>\n"
                "[2026-11-21 04:00:03] {http.py:75} INFO - Calling GET https://jsonplaceholder.typicode.com/users\n"
                "[2026-11-21 04:00:05] {http.py:92} ERROR - HTTP error status: 500 Server Error for url.\n"
                "requests.exceptions.HTTPError: 500 Server Error: Internal Server Error for url\n"
                "[2026-11-21 04:00:06] {taskinstance.py:1463} ERROR - Task failed with exception."
            ),
            "PostgreSQL Connection Pool Exhaustion (max_connections reached)": (
                "[2026-11-22 04:05:01] {taskinstance.py:1150} INFO - Executing <Task(PostgresOperator): load_to_postgres>\n"
                "[2026-11-22 04:05:02] {postgres.py:54} INFO - Connecting to PostgreSQL warehouse at 5432/analytics...\n"
                "[2026-11-22 04:05:04] {postgres.py:68} ERROR - FATAL: remaining connection slots are reserved for non-replication superuser connections\n"
                "psycopg2.OperationalError: FATAL: remaining connection slots are reserved for non-replication superuser connections\n"
                "[2026-11-22 04:05:05] {taskinstance.py:1463} ERROR - Task failed with exception."
            ),
            "Amazon S3 Sensor Timeout (AirflowSensorTimeout: Missing partition)": (
                "[2026-11-23 04:01:00] {taskinstance.py:1150} INFO - Executing <Task(S3KeySensor): wait_for_daily_lake_partition>\n"
                "[2026-11-23 04:01:01] {s3.py:44} INFO - Poking s3a://ayush-consumer-producer/data/sample/2026-11-23.json\n"
                "[2026-11-23 04:06:01] {sensor.py:102} ERROR - AirflowSensorTimeout: Sensor has timed out after 300 seconds.\n"
                "[2026-11-23 04:06:02] {taskinstance.py:1463} ERROR - Task failed with AirflowSensorTimeout."
            ),
            "PySpark Data Quality Rule Breach (Null Identity Email)": (
                "[2026-11-24 04:03:10] {taskinstance.py:1150} INFO - Executing <Task(PythonOperator): audit_data_quality>\n"
                "[2026-11-24 04:03:12] {data_quality.py:35} INFO - Inspecting 10 user rows against schema StructType...\n"
                "[2026-11-24 04:03:13] {data_quality.py:48} ERROR - Data quality check failed: Primary identity column 'email' has 3 NULL records!\n"
                "ValueError: Data quality check failed: Primary key violation.\n"
                "[2026-11-24 04:03:14] {taskinstance.py:1463} ERROR - Task aborted by quality gate."
            ),
            "Nominal Healthy Execution Trace": (
                "[2026-11-25 04:00:01] {taskinstance.py:1150} INFO - Executing <Task(ProducerDAG): extract_and_stage>\n"
                "[2026-11-25 04:00:02] {http.py:75} INFO - HTTP GET returned 200 OK. 10 records extracted in 0.4s.\n"
                "[2026-11-25 04:00:03] {s3.py:50} INFO - Staged 10 records to s3a://ayush-consumer-producer/data/sample/2026-11-25.json\n"
                "[2026-11-25 04:00:05] {spark.py:80} INFO - PySpark schema validated: 0 null records. 100% data quality score.\n"
                "[2026-11-25 04:00:06] {postgres.py:90} INFO - Task consumer_users_daily.load_to_postgres completed with state SUCCESS\n"
                "[2026-11-25 04:00:07] {dagrun.py:320} INFO - Overall Status: SUCCESS. Pipeline SLA respected."
            )
        }
        
        input_log = ""
        if log_entry_mode == "Select from Realistic Error Presets":
            log_sample_choice = st.selectbox(
                "Select an Airflow Execution Log Preset to Diagnose:",
                options=list(preset_logs.keys())
            )
            input_log = preset_logs[log_sample_choice]
        elif log_entry_mode == "Paste Custom Raw Log Text":
            input_log = st.text_area(
                "Paste or Type Any Airflow Task Execution Log Below:",
                value=(
                    "[2026-11-25 04:02:28] {spark_submit.py:180} ERROR - java.lang.OutOfMemoryError: Java heap space\n"
                    "  at org.apache.spark.memory.UnifiedMemoryManager.acquireExecutionMemory(UnifiedMemoryManager.scala:132)\n"
                    "  at org.apache.spark.sql.execution.aggregate.HashAggregateExec.doExecute(HashAggregateExec.scala:114)\n"
                    "[2026-11-25 04:02:29] {taskinstance.py:1463} ERROR - Task failed with exception."
                ),
                height=180
            )
        else:
            historical_keys = list(logs_dict.keys())
            selected_key = st.selectbox("Select Historical Log ID to Query:", options=historical_keys[:50])
            input_log = logs_dict[selected_key]
            
        if input_log:
            diag = parse_log_root_cause(input_log)
            
            st.markdown("<div class='section-title'>Automated Log Diagnosis & Remediation</div>", unsafe_allow_html=True)
            col_d1, col_d2 = st.columns([1, 1.3])
            
            with col_d1:
                st.markdown(f"""
                <div class='diagnostic-panel'>
                    <strong>Identified Component Stage:</strong> <code>{diag['stage']}</code><br><br>
                    <strong>Detected Exception Signature:</strong> <code>{diag['error_signature'] or 'None (Clean Run)'}</code><br><br>
                    <strong>Plain-English Root Cause:</strong><br>
                    {diag['root_cause']}<br><br>
                    <strong>Actionable Engineering Remediation:</strong><br>
                    {diag['recommendation']}
                </div>
                """, unsafe_allow_html=True)
                
            with col_d2:
                st.caption("Active Airflow Task Log Trace:")
                st.markdown(f"<div class='log-container'>{input_log.replace('<', '&lt;').replace('>', '&gt;')}</div>", unsafe_allow_html=True)

# =============================================================================
# TAB 3: Airflow Pipeline Inspector (Animated Interactive Flow)
# =============================================================================
with tab3:
    st.subheader("Data Engineering Pipeline Architecture & Live Execution")
    

    
    # Interactive Operational Simulator Picker
    st.markdown("<div class='section-title'>1. Interactive Pipeline Execution Flow (Live Animation)</div>", unsafe_allow_html=True)
    st.markdown("Select an operational scenario to watch the live data flow across all 4 stages in real time:")
    
    sim_scenario = st.selectbox(
        "Choose Operational Scenario to Simulate:",
        options=[
            "Nominal Operational Flow (Healthy End-to-End Execution)",
            "Simulate Upstream Failure: REST API Returns 500 Internal Error",
            "Simulate Corrupted Data: PySpark Catches Null Identity Values",
            "Simulate Database Lock: PostgreSQL Connection Pool Exhaustion"
        ]
    )
    
    # Configure animated node classes based on scenario
    if "Nominal" in sim_scenario:
        n1_cls, n2_cls, n3_cls, n4_cls = "node-active-green", "node-active-green", "node-active-green", "node-active-green"
        p1_cls, p2_cls, p3_cls = "pipe-active-green", "pipe-active-green", "pipe-active-green"
        s1_tag, s2_tag, s3_tag, s4_tag = "ACTIVE (200 OK)", "STAGED (S3 PUT)", "AUDITED (0 NULLS)", "COMMITTED (10 ROWS)"
        hud_msg = "Pipeline Status: End-to-End Execution Passed within SLA (1.2 seconds)."
        hud_type = "success"
    elif "API Returns 500" in sim_scenario:
        n1_cls, n2_cls, n3_cls, n4_cls = "node-active-danger", "node-skipped", "node-skipped", "node-skipped"
        p1_cls, p2_cls, p3_cls = "pipe-blocked", "pipe-connector", "pipe-connector"
        s1_tag, s2_tag, s3_tag, s4_tag = "FAILED (HTTP 500)", "SKIPPED", "SKIPPED", "SKIPPED"
        hud_msg = "Pipeline Blocked at Stage 1: Upstream server crash detected. Model flags API_INGESTION as root cause."
        hud_type = "error"
    elif "PySpark Catches Null" in sim_scenario:
        n1_cls, n2_cls, n3_cls, n4_cls = "node-active-green", "node-active-green", "node-active-warn", "node-skipped"
        p1_cls, p2_cls, p3_cls = "pipe-active-green", "pipe-active-green", "pipe-blocked"
        s1_tag, s2_tag, s3_tag, s4_tag = "PASSED (200 OK)", "STAGED (RAW JSON)", "FAILED (NULL DETECTED)", "BLOCKED (SAFETY)"
        hud_msg = "Quality Gate Interlock Active: Corrupt records quarantined to prevent database pollution. Root cause: SPARK_PROCESSING."
        hud_type = "warning"
    else:
        n1_cls, n2_cls, n3_cls, n4_cls = "node-active-green", "node-active-green", "node-active-green", "node-active-danger"
        p1_cls, p2_cls, p3_cls = "pipe-active-green", "pipe-active-green", "pipe-blocked"
        s1_tag, s2_tag, s3_tag, s4_tag = "PASSED (200 OK)", "STAGED (RAW JSON)", "AUDITED (0 NULLS)", "FAILED (POOL FULL)"
        hud_msg = "Warehouse Connection Failure: max_connections=100 reached. Attention-GRU attributes POSTGRES_DATABASE."
        hud_type = "error"

    # Render Animated Visual Pipeline Canvas
    st.markdown(f"""
    <div class='pipeline-canvas'>
        <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;'>
            <span style='font-size: 0.82rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: #94A3B8;'>Airflow DAG Orchestration Channel</span>
            <span style='font-size: 0.78rem; font-family: "JetBrains Mono", monospace; color: #38BDF8;'>Schedule: @daily | Zero Lookahead Isolation</span>
        </div>
        <div class='pipeline-nodes-wrapper'>
            <!-- Node 1 -->
            <div class='stage-node-box {n1_cls}'>
                <div style='font-size: 0.70rem; font-weight: 700; color: #94A3B8; text-transform: uppercase;'>Stage 1</div>
                <div style='font-size: 0.90rem; font-weight: 700; margin: 4px 0;'>REST API Ingestion</div>
                <div style='font-size: 0.72rem; color: #CBD5E1;'>HTTP GET :80/443</div>
                <div style='margin-top: 8px; font-size: 0.75rem; font-weight: 600;'>{s1_tag}</div>
            </div>
            <!-- Pipe 1-2 -->
            <div class='pipe-connector {p1_cls}'></div>
            <!-- Node 2 -->
            <div class='stage-node-box {n2_cls}'>
                <div style='font-size: 0.70rem; font-weight: 700; color: #94A3B8; text-transform: uppercase;'>Stage 2</div>
                <div style='font-size: 0.90rem; font-weight: 700; margin: 4px 0;'>Amazon S3 Lake</div>
                <div style='font-size: 0.72rem; color: #CBD5E1;'>s3a:// Partition Vault</div>
                <div style='margin-top: 8px; font-size: 0.75rem; font-weight: 600;'>{s2_tag}</div>
            </div>
            <!-- Pipe 2-3 -->
            <div class='pipe-connector {p2_cls}'></div>
            <!-- Node 3 -->
            <div class='stage-node-box {n3_cls}'>
                <div style='font-size: 0.70rem; font-weight: 700; color: #94A3B8; text-transform: uppercase;'>Stage 3</div>
                <div style='font-size: 0.90rem; font-weight: 700; margin: 4px 0;'>PySpark Quality Audit</div>
                <div style='font-size: 0.72rem; color: #CBD5E1;'>Distributed StructType</div>
                <div style='margin-top: 8px; font-size: 0.75rem; font-weight: 600;'>{s3_tag}</div>
            </div>
            <!-- Pipe 3-4 -->
            <div class='pipe-connector {p3_cls}'></div>
            <!-- Node 4 -->
            <div class='stage-node-box {n4_cls}'>
                <div style='font-size: 0.70rem; font-weight: 700; color: #94A3B8; text-transform: uppercase;'>Stage 4</div>
                <div style='font-size: 0.90rem; font-weight: 700; margin: 4px 0;'>PostgreSQL Warehouse</div>
                <div style='font-size: 0.72rem; color: #CBD5E1;'>Port 5432 / Idempotent</div>
                <div style='margin-top: 8px; font-size: 0.75rem; font-weight: 600;'>{s4_tag}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if hud_type == "success":
        st.success(hud_msg)
    elif hud_type == "warning":
        st.warning(hud_msg)
    else:
        st.error(hud_msg)

    # 4 Pipeline Stages Deep-Dive
    st.markdown("<div class='section-title'>2. The 4 Pipeline Stages (Click to Explore Architecture Details)</div>", unsafe_allow_html=True)
    
    selected_stage = st.radio(
        "Select Pipeline Stage to Inspect Subsystem Details:",
        options=[
            "Stage 1: Upstream REST API Ingestion",
            "Stage 2: Amazon S3 Cloud Lake Staging",
            "Stage 3: PySpark Distributed Quality Audit",
            "Stage 4: PostgreSQL Idempotent Load"
        ],
        horizontal=True
    )
    
    col_st1, col_st2 = st.columns([1.2, 1])
    
    if "Stage 1" in selected_stage:
        with col_st1:
            st.markdown(f"""
            <div class='step-card'>
                <div class='step-card-num'>Pipeline Stage 1 of 4</div>
                <div class='step-card-title'>Upstream REST API Ingestion (Producer DAG)</div>
                <div class='step-card-desc'>
                    • <strong>Plain English Role:</strong> Connects to external servers over the internet to extract daily user records.<br>
                    • <strong>Production DAG:</strong> <code>producer_users_daily.py</code><br>
                    • <strong>Real Source Endpoint:</strong> <code>https://jsonplaceholder.typicode.com/users</code><br>
                    • <strong>Protocol:</strong> HTTP GET with 10-second connection timeout.<br>
                    • <strong>Failure Modes Detected by our Model:</strong> API Server Down (HTTP 500), Rate Limit Quota Reached (HTTP 429), or Network Gateway Timeout (ReadTimeout).
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col_st2:
            st.markdown("""
            <div class='diagnostic-panel'>
                <strong>Sample Ingested JSON Payload:</strong><br>
                <code style='font-size: 0.78rem;'>
                &#123;<br>
                &nbsp;&nbsp;"id": 1,<br>
                &nbsp;&nbsp;"name": "Leanne Graham",<br>
                &nbsp;&nbsp;"email": "Sincere@april.biz"<br>
                &#125;
                </code>
            </div>
            """, unsafe_allow_html=True)
            
    elif "Stage 2" in selected_stage:
        with col_st1:
            st.markdown(f"""
            <div class='step-card'>
                <div class='step-card-num'>Pipeline Stage 2 of 4</div>
                <div class='step-card-title'>Cloud Data Lake Staging (Amazon S3)</div>
                <div class='step-card-desc'>
                    • <strong>Plain English Role:</strong> Acts as an immutable storage vault where raw incoming files are archived before transformation.<br>
                    • <strong>Bucket Path:</strong> <code>s3a://ayush-consumer-producer/data/sample/YYYY-MM-DD.json</code><br>
                    • <strong>Sensor Synchronization:</strong> <code>S3KeySensor</code> periodically checks ("pokes") for the existence of today's file before allowing downstream jobs to run.<br>
                    • <strong>Failure Modes Detected:</strong> Missing S3 partition key (upstream task was skipped), Access Denied (IAM role credentials expired), or 0-byte empty file.
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col_st2:
            st.markdown("""
            <div class='diagnostic-panel'>
                <strong>Storage Architecture Properties:</strong><br>
                • <strong>Bucket:</strong> <code>ayush-consumer-producer</code><br>
                • <strong>Storage Class:</strong> S3 Standard (Hot Tier)<br>
                • <strong>Partition Format:</strong> Date-partitioned JSON<br>
                • <strong>Sensor Poke Interval:</strong> 60 seconds (Timeout: 300s)
            </div>
            """, unsafe_allow_html=True)
            
    elif "Stage 3" in selected_stage:
        with col_st1:
            st.markdown(f"""
            <div class='step-card'>
                <div class='step-card-num'>Pipeline Stage 3 of 4</div>
                <div class='step-card-title'>PySpark Distributed Transformation & Quality Audit</div>
                <div class='step-card-desc'>
                    • <strong>Plain English Role:</strong> High-performance distributed data filter that ensures only 100% clean, valid records enter the database.<br>
                    • <strong>Engine:</strong> Apache Spark (PySpark) with strict <code>StructType</code> schema enforcement.<br>
                    • <strong>Quality Validation Rule:</strong> Rejects batches if primary identity columns (<code>id</code>, <code>email</code>) contain NULL or corrupted data types.<br>
                    • <strong>Failure Modes Detected:</strong> Schema drift (source added incompatible columns), Null value breach, or Spark Executor JVM Out-Of-Memory (OOM).
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col_st2:
            st.markdown("""
            <div class='diagnostic-panel'>
                <strong>Data Quality Assurance Metrics:</strong><br>
                • <strong>Required Schema:</strong> <code>id: INT, name: STR, email: STR</code><br>
                • <strong>Null Tolerance:</strong> 0% (Strict zero-defect gate)<br>
                • <strong>Processing Framework:</strong> PySpark Local/Cluster Session<br>
                • <strong>Output Staging:</strong> Parquet clean partition
            </div>
            """, unsafe_allow_html=True)
            
    else:
        with col_st1:
            st.markdown(f"""
            <div class='step-card'>
                <div class='step-card-num'>Pipeline Stage 4 of 4</div>
                <div class='step-card-title'>PostgreSQL Relational Warehouse Load</div>
                <div class='step-card-desc'>
                    • <strong>Plain English Role:</strong> Final destination where analytics tables live for business reporting and dashboards.<br>
                    • <strong>Target Table:</strong> <code>users_daily (id, name, email, dt)</code><br>
                    • <strong>Idempotent Safety:</strong> Uses <code>DELETE FROM users_daily WHERE dt = 'today'</code> followed by <code>INSERT</code>. If the pipeline re-runs 10 times, the database remains 100% accurate with zero duplicate rows.<br>
                    • <strong>Failure Modes Detected:</strong> Port 5432 Connection Refused, Connection Pool Exhaustion (max_connections reached), or Table Deadlock timeouts.
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col_st2:
            st.markdown("""
            <div class='diagnostic-panel'>
                <strong>Database Transaction Properties:</strong><br>
                • <strong>Engine:</strong> PostgreSQL 14 (ACID Compliant)<br>
                • <strong>Strategy:</strong> Partition Delete-and-Insert<br>
                • <strong>Transaction Isolation:</strong> Read Committed<br>
                • <strong>Connection Management:</strong> Airflow PostgresHook
            </div>
            """, unsafe_allow_html=True)

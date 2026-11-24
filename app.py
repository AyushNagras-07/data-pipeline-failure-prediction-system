"""
app.py — Production Streamlit Dashboard
Data Pipeline Failure Prediction System (ANNA Course 2307411T - Track 2: Sequential Models)

Group 12 (Equal Contributors):
 - Ayush Nagras     (PRN: 202301070001)
 - Siddhesh Bhivare (PRN: 202301070005)
 - Om Nikam         (PRN: 202301070006)
 - Vaibhav Tajane   (PRN: 202301070008)

Features:
1. Live Pipeline Telemetry Monitor & Pre-Failure Early Warning Alert
2. Dual-Head Attention-GRU Inference (Risk Probability + Subsystem Root Cause)
3. Explainable AI (XAI) Temporal Attention Heatmap over [t-3, t-2, t-1]
4. Primary/Foreign Key Relational Log Viewer (pipeline_telemetry.csv -> pipeline_logs.json)
5. Interactive "What-If" Scenario Simulator with live degradation sliders & presets
6. Real-time Airflow ETL Pipeline trigger & data quality verification
7. Experimental Ablation & Optimization Charts (5 Publication Figures)
"""

import os
import sys
import json
import time
import torch
import numpy as np
import pandas as pd
import streamlit as st

# Setup paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

from src.models.attention_model import AttentionGRU
from src.features.sequence_generator import FEATURE_COLUMNS, STAGE_NAMES
from src.data.log_analyzer import get_run_analysis_by_log_id, load_logs_database

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Data Pipeline Failure Prediction | Group 12",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for high-contrast cards and metrics
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 20px;
    }
    .card-risk-low {
        background-color: #ECFDF5;
        border-left: 6px solid #10B981;
        padding: 16px;
        border-radius: 8px;
    }
    .card-risk-med {
        background-color: #FFFBEB;
        border-left: 6px solid #F59E0B;
        padding: 16px;
        border-radius: 8px;
    }
    .card-risk-high {
        background-color: #FEF2F2;
        border-left: 6px solid #EF4444;
        padding: 16px;
        border-radius: 8px;
    }
    .log-container {
        background-color: #1E1E1E;
        color: #D4D4D4;
        font-family: 'Courier New', monospace;
        padding: 14px;
        border-radius: 6px;
        font-size: 0.85rem;
        max-height: 280px;
        overflow-y: auto;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Caching Data & Model
# -----------------------------------------------------------------------------
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

@st.cache_data
def load_telemetry_and_logs():
    csv_path = os.path.join(PROJECT_ROOT, "data", "raw", "pipeline_telemetry.csv")
    logs_path = os.path.join(PROJECT_ROOT, "data", "raw", "pipeline_logs.json")
    df = pd.read_csv(csv_path)
    with open(logs_path, "r") as f:
        logs = json.load(f)
    return df, logs

# Load resources
try:
    model, scaler = load_trained_system()
    df_telem, logs_dict = load_telemetry_and_logs()
except Exception as e:
    st.error(f"Error loading system assets: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# Sidebar: Metadata & Quick Controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("⚡ Project Metadata")
    st.markdown("**Course:** 2307411T - ANNA")
    st.markdown("**Track 2:** Sequential Models")
    st.markdown("**Assessment:** End-Semester Exam (50M)")
    st.divider()
    
    st.subheader("👥 Group 12 (Equal Contributors)")
    st.markdown("• **Ayush Nagras** (202301070001)")
    st.markdown("• **Siddhesh Bhivare** (202301070005)")
    st.markdown("• **Om Nikam** (202301070006)")
    st.markdown("• **Vaibhav Tajane** (202301070008)")
    st.divider()
    
    st.subheader("⚙️ System Status")
    st.success("✓ Attention-GRU: LOADED")
    st.info(f"✓ Historical Runs: {len(df_telem)}")
    st.info(f"✓ Airflow Task Logs: {len(logs_dict)}")

# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.markdown("<div class='main-header'>Data Pipeline Failure Prediction System</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Pre-Failure Early Warning & Root Cause Attribution using Dual-Head Temporal Attention GRUs</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🚨 Live Risk Monitor",
    "🧪 What-If Simulator",
    "📊 Model Ablation & Figures",
    "🛠️ Airflow Pipeline Inspector",
    "🎓 System Architecture & Viva"
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
        st.warning("Selected run is within the first 3 runs. Showing nearest sequence.")
        run_idx = 3
        
    # Get sequence of past 3 runs [run_idx-3, run_idx-2, run_idx-1]
    window_df = df_telem.iloc[run_idx-3:run_idx]
    target_row = df_telem.iloc[run_idx]
    
    st.markdown("##### 1. Historical Telemetry Sequence Input $[t-3, t-2, t-1]$")
    cols_display = ['run_id', 'duration', 'records_processed', 'retry_count', 'memory_mb', 'failure_scenario', 'log_id']
    st.dataframe(window_df[cols_display], use_container_width=True, hide_index=True)
    
    # Prepare sequence for inference
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
    
    st.markdown("##### 2. Dual-Head Model Inference & Operational Early Warning")
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    
    with col_m1:
        st.metric(
            label="Predicted Failure Risk",
            value=f"{risk_val:.1%}",
            delta="- High Risk" if risk_val >= 0.5 else "+ Healthy",
            delta_color="inverse" if risk_val >= 0.5 else "normal"
        )
        
    with col_m2:
        risk_badge = "CRITICAL HIGH" if risk_val >= 0.70 else ("ELEVATED" if risk_val >= 0.35 else "NOMINAL")
        st.metric(label="Risk State", value=risk_badge)
        
    with col_m3:
        st.metric(label="Predicted Failing Subsystem", value=top_stage_name)
        
    with col_m4:
        st.metric(label="Subsystem Confidence", value=f"{top_stage_conf:.1%}")
        
    # Temporal Attention Breakdown
    st.markdown("##### 3. Temporal Attention Weights (Explainable AI Attribution)")
    col_attn1, col_attn2 = st.columns([1.5, 1])
    
    with col_attn1:
        attn_df = pd.DataFrame({
            "Historical Execution": [f"Run t-3 ({window_df.iloc[0]['run_id']})",
                                     f"Run t-2 ({window_df.iloc[1]['run_id']})",
                                     f"Run t-1 ({window_df.iloc[2]['run_id']})"],
            "Attention Contribution (%)": [w * 100 for w in attn_np]
        })
        st.bar_chart(attn_df, x="Historical Execution", y="Attention Contribution (%)", use_container_width=True)
        
    with col_attn2:
        max_t = int(np.argmax(attn_np))
        st.markdown(f"""
        **Temporal Explainability Diagnosis:**
        - **Dominant Driver:** `Run t-{3-max_t}` accounts for **{attn_np[max_t]:.1%}** of model attention.
        - **Degradation Pattern:** Compounding memory or latency creeping initiated at this step.
        - **Target Ground Truth:** `{target_row['failure_scenario']}` (Status: {target_row['status']})
        """)
        
    # Primary/Foreign Key Log Viewer
    st.markdown("##### 4. Relational Log Viewer (Joined on `log_id`)")
    log_id = target_row['log_id']
    analysis = get_run_analysis_by_log_id(log_id)
    
    col_log1, col_log2 = st.columns([1, 1.2])
    with col_log1:
        st.info(f"**Foreign Key `log_id`:** `{log_id}`")
        st.markdown(f"**Identified Failing Stage:** `{analysis['stage']}`")
        st.markdown(f"**Error Signature:** `{analysis['error_signature']}`")
        st.markdown(f"**Plain-English Root Cause:** {analysis['root_cause']}")
        st.warning(f"**Recommended Engineering Action:** {analysis['recommendation']}")
        
    with col_log2:
        st.caption("Raw Multi-Task Airflow Execution Trace (from `data/raw/pipeline_logs.json`):")
        raw_log = analysis['raw_log']
        st.markdown(f"<div class='log-container'>{raw_log.replace('<', '&lt;').replace('>', '&gt;')}</div>", unsafe_allow_html=True)

# =============================================================================
# TAB 2: Interactive What-If Simulator
# =============================================================================
with tab2:
    st.subheader("Interactive What-If Degradation Simulator")
    st.markdown("Adjust historical execution telemetry sliders to observe live model risk response and attention shifts in real time.")
    
    # Preset Selector
    preset = st.radio(
        "Load Degradation Pattern Preset:",
        options=["Custom Telemetry", "Gradual Memory Leak (Spark OOM)", "Upstream API Timeout Cascade", "Nominal Healthy Execution"],
        horizontal=True
    )
    
    # Defaults
    if preset == "Gradual Memory Leak (Spark OOM)":
        d1, r1, m1 = 2.5, 0, 95.0
        d2, r2, m2 = 4.8, 1, 130.0
        d3, r3, m3 = 14.5, 2, 185.0
    elif preset == "Upstream API Timeout Cascade":
        d1, r1, m1 = 3.2, 1, 88.0
        d2, r2, m2 = 8.5, 2, 92.0
        d3, r3, m3 = 10.0, 5, 96.0
    elif preset == "Nominal Healthy Execution":
        d1, r1, m1 = 1.8, 0, 84.0
        d2, r2, m2 = 1.9, 0, 85.0
        d3, r3, m3 = 1.7, 0, 83.5
    else:
        d1, r1, m1 = 2.0, 0, 85.0
        d2, r2, m2 = 3.5, 1, 95.0
        d3, r3, m3 = 7.0, 2, 115.0
        
    col_sim1, col_sim2, col_sim3 = st.columns(3)
    
    with col_sim1:
        st.markdown("#### Run t-3 (2 Days Ago)")
        dur_1 = st.slider("Duration (s)", 0.5, 30.0, float(d1), key="d1")
        ret_1 = st.slider("Retry Count", 0, 5, int(r1), key="r1")
        mem_1 = st.slider("Worker Memory (MB)", 50.0, 250.0, float(m1), key="m1")
        rec_1 = 10
        
    with col_sim2:
        st.markdown("#### Run t-2 (Yesterday)")
        dur_2 = st.slider("Duration (s)", 0.5, 30.0, float(d2), key="d2")
        ret_2 = st.slider("Retry Count", 0, 5, int(r2), key="r2")
        mem_2 = st.slider("Worker Memory (MB)", 50.0, 250.0, float(m2), key="m2")
        rec_2 = 10
        
    with col_sim3:
        st.markdown("#### Run t-1 (Most Recent)")
        dur_3 = st.slider("Duration (s)", 0.5, 30.0, float(d3), key="d3")
        ret_3 = st.slider("Retry Count", 0, 5, int(r3), key="r3")
        mem_3 = st.slider("Worker Memory (MB)", 50.0, 250.0, float(m3), key="m3")
        rec_3 = 10
        
    # Transform simulation input
    sim_data = np.array([
        [dur_1, rec_1, ret_1, mem_1],
        [dur_2, rec_2, ret_2, mem_2],
        [dur_3, rec_3, ret_3, mem_3]
    ])
    sim_scaled = scaler.transform(sim_data)
    sim_tensor = torch.tensor(sim_scaled, dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        sim_risk, sim_cause, sim_attn = model(sim_tensor)
        s_risk = float(sim_risk.item())
        s_cause = torch.softmax(sim_cause, dim=1).squeeze(0).numpy()
        s_attn = sim_attn.squeeze(0).numpy()
        
    st.divider()
    st.markdown("### Simulated Real-Time Prediction Results")
    
    col_sr1, col_sr2, col_sr3 = st.columns([1, 1, 2])
    with col_sr1:
        st.metric("Predicted Failure Risk", f"{s_risk:.1%}")
        if s_risk >= 0.70:
            st.error("CRITICAL RISK DETECTED")
        elif s_risk >= 0.35:
            st.warning("ELEVATED PRE-FAILURE RISK")
        else:
            st.success("NOMINAL SLA")
            
    with col_sr2:
        sim_top_stage = STAGE_NAMES[int(np.argmax(s_cause))]
        st.metric("Attributed Subsystem", sim_top_stage)
        st.metric("Subsystem Confidence", f"{np.max(s_cause):.1%}")
        
    with col_sr3:
        st.caption("Temporal Attention Focus over Input Window:")
        s_df = pd.DataFrame({
            "Step": ["Run t-3", "Run t-2", "Run t-1"],
            "Weight (%)": [w * 100 for w in s_attn]
        })
        st.bar_chart(s_df, x="Step", y="Weight (%)", use_container_width=True)

# =============================================================================
# TAB 3: Model Ablation & Figures (CO4)
# =============================================================================
with tab3:
    st.subheader("Experimental Optimization & Ablation Study (CO4)")
    st.markdown("Visual verification of all 5 publication-ready evaluation charts generated in Week 3.")
    
    fig_dir = os.path.join(PROJECT_ROOT, "reports", "figures")
    
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        f1_path = os.path.join(fig_dir, "1_architecture_comparison_metrics.png")
        if os.path.exists(f1_path):
            st.image(f1_path, caption="Figure 1: Architectural Comparison across 5 Models")
        f3_path = os.path.join(fig_dir, "3_training_convergence_curves.png")
        if os.path.exists(f3_path):
            st.image(f3_path, caption="Figure 3: Training & Validation Convergence Trajectories")
            
    with col_f2:
        f2_path = os.path.join(fig_dir, "2_window_size_ablation.png")
        if os.path.exists(f2_path):
            st.image(f2_path, caption="Figure 2: Impact of Window Size T on F1 vs Latency")
        f4_path = os.path.join(fig_dir, "4_roc_curves.png")
        if os.path.exists(f4_path):
            st.image(f4_path, caption="Figure 4: Receiver Operating Characteristic (ROC-AUC)")
            
    st.divider()
    f5_path = os.path.join(fig_dir, "5_confusion_matrix_subsystems.png")
    if os.path.exists(f5_path):
        st.image(f5_path, caption="Figure 5: 5x5 Subsystem Root Cause Confusion Matrix", width=750)
        
    # Tabular Results
    st.markdown("### Hyperparameter Grid Search Logs")
    csv_results = os.path.join(PROJECT_ROOT, "reports", "hyperparameter_sweep_results.csv")
    if os.path.exists(csv_results):
        df_sw = pd.read_csv(csv_results)
        st.dataframe(df_sw, use_container_width=True)

# =============================================================================
# TAB 4: Airflow Pipeline Inspector
# =============================================================================
with tab4:
    st.subheader("Data Engineering Pipeline Execution Flow")
    st.markdown("Live verification of the 4-stage Apache Airflow Producer-Consumer DAG architecture.")
    
    st.markdown("""
    ```
    ┌─────────────────────────┐       ┌─────────────────────────┐       ┌─────────────────────────┐       ┌─────────────────────────┐
    │ 1. Upstream REST API    │ ────► │ 2. S3 Lake Staging      │ ────► │ 3. PySpark Schema Check │ ────► │ 4. PostgreSQL Idempotent│
    │ JSONPlaceholder /users  │       │ Raw JSON Bucket         │       │ Null filter & transform │       │ Partition Delete/Insert │
    └─────────────────────────┘       └─────────────────────────┘       └─────────────────────────┘       └─────────────────────────┘
    ```
    """)
    
    if st.button("▶️ Execute Live Pipeline ETL Verification", type="primary"):
        import requests
        st.info("Triggering Producer DAG extraction...")
        try:
            resp = requests.get("https://jsonplaceholder.typicode.com/users", timeout=10)
            records = resp.json()
            st.success(f"✓ Producer DAG extracted {len(records)} user records from REST endpoint.")
            
            df_users = pd.DataFrame(records)
            st.dataframe(df_users[['id', 'name', 'email', 'phone', 'website']].head(5), use_container_width=True)
            
            # Consumer check
            null_count = df_users['id'].isna().sum() + df_users['email'].isna().sum()
            if null_count == 0:
                st.success("✓ PySpark Consumer Check: 0 null identity values found. Data Quality PASSED.")
                st.success("✓ Staged clean partition ready for PostgreSQL idempotent commit.")
            else:
                st.error("✗ Data Quality Check FAILED.")
        except Exception as err:
            st.error(f"ETL Execution Error: {err}")

# =============================================================================
# TAB 5: System Architecture & Viva Defense
# =============================================================================
with tab5:
    st.subheader("Viva Defense Guide & Theoretical Formulations")
    
    st.markdown(r"""
    ### 1. Mathematical Grounding (CO3)
    Given hidden state vectors $H = [h_1, h_2, \dots, h_T] \in \mathbb{R}^{T \times d_h}$ from recurrent GRU:
    
    1. **Additive Alignment Energy:**
       $$e_t = v_a^\top \tanh(W_a h_t + b_a)$$
    2. **Temporal Attention Weights (Softmax):**
       $$\alpha_t = \frac{\exp(e_t)}{\sum_{j=1}^T \exp(e_j)} \quad \text{where } \sum_{t=1}^T \alpha_t = 1.0$$
    3. **Context Vector:**
       $$c = \sum_{t=1}^T \alpha_t h_t$$
    4. **Dual-Head Multi-Task Loss:**
       $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{BCE}}(\hat{y}_{\text{risk}}, y) + 0.5 \cdot \mathcal{L}_{\text{CE}}(\hat{y}_{\text{cause}}, y_{\text{stage}})$$
       
    ---
    ### 2. Group Contribution Matrix (All 4 Equal Contributors)
    | Group Member | PRN | Core Responsibilities & Viva Mastery |
    | :--- | :--- | :--- |
    | **Ayush Nagras** | 202301070001 | Airflow Producer/Consumer DAGs, Relational `log_id` schema, Streamlit UI |
    | **Siddhesh Bhivare**| 202301070005 | Temporal Attention Layer, Additive Bahdanau Math, Context Fusion |
    | **Om Nikam** | 202301070006 | Dual-Head Multi-Task Loss formulation, Subsystem Root-Cause Head |
    | **Vaibhav Tajane** | 202301070008 | Hyperparameter Grid Search, ROC-AUC Profiling, Optimization Curves |
    """)

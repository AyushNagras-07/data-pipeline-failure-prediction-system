"""
attention_visualizer.py
Explainable AI (XAI) utilities for Temporal Attention Mechanisms.
Provides tabular, visual, and textual breakdowns of attention weights across historical runs
to demonstrate which telemetry signals triggered the model's prediction.
"""

import numpy as np

def make_ascii_bar(val, max_val=1.0, length=20):
    """Generates an ASCII progress bar representing an attention weight."""
    filled = int(round((val / max_val) * length))
    filled = max(0, min(length, filled))
    return "█" * filled + "░" * (length - filled)

def format_attention_report(attn_weights, window_indices=None, past_telemetry=None):
    """
    Formats the attention weights across the temporal sliding window [t-W, ..., t-1]
    into an interpretable diagnostic breakdown.
    
    Parameters:
      attn_weights: list or 1D array of length W
      window_indices: list of labels, e.g. ['t-3', 't-2', 't-1']
      past_telemetry: optional list of dicts with telemetry stats for each past run
    """
    W = len(attn_weights)
    if window_indices is None:
        window_indices = [f"Run t-{W - i}" for i in range(W)]
        
    lines = []
    lines.append(" Temporal Attention Distribution (Temporal Causality Breakdown):")
    lines.append(" " + "-" * 72)
    lines.append(f" {'Time-Step':<12} | {'Weight':<8} | {'Attention Visual':<22} | {'Observed Telemetry Summary'}")
    lines.append(" " + "-" * 72)
    
    for i in range(W):
        w = float(attn_weights[i])
        bar = make_ascii_bar(w, max_val=1.0, length=20)
        label = window_indices[i]
        
        telem_str = "Standard operational profile"
        if past_telemetry and i < len(past_telemetry):
            t = past_telemetry[i]
            dur = t.get('duration', '-')
            retries = t.get('retry_count', '-')
            mem = t.get('memory_mb', '-')
            telem_str = f"Dur: {dur}s | Retries: {retries} | Mem: {mem}MB"
            
        lines.append(f" {label:<12} | {w:>6.1%}  | {bar:<22} | {telem_str}")
        
    lines.append(" " + "-" * 72)
    
    # Identify dominant temporal driver
    max_idx = int(np.argmax(attn_weights))
    max_w = attn_weights[max_idx]
    lines.append(f" Dominant Temporal Driver: {window_indices[max_idx]} accounts for {max_w:.1%} of model attention.")
    return "\n".join(lines)

def generate_prediction_narrative(risk_prob, cause_probs, attn_weights, stage_names, window_meta=None):
    """
    Synthesizes a plain-English explainability narrative combining risk, root cause, and temporal attention.
    """
    W = len(attn_weights)
    risk_level = "CRITICAL HIGH" if risk_prob >= 0.70 else ("ELEVATED" if risk_prob >= 0.35 else "NOMINAL")
    
    top_stage_idx = int(np.argmax(cause_probs))
    top_stage = stage_names[top_stage_idx]
    top_stage_conf = cause_probs[top_stage_idx]
    
    max_attn_idx = int(np.argmax(attn_weights))
    step_label = f"Run t-{W - max_attn_idx}"
    
    narrative = []
    narrative.append(f"• Operational Risk State: [{risk_level}] (Failure Probability: {risk_prob:.1%})")
    if risk_prob >= 0.35:
        narrative.append(f"• Predicted Failing Subsystem: {top_stage} ({top_stage_conf:.1%} model confidence)")
        narrative.append(f"• Temporal Root-Cause Anchor: {step_label} contributed {attn_weights[max_attn_idx]:.1%} of the model's focus,")
        narrative.append(f"  indicating that pre-failure degradation initiated during {step_label}.")
    else:
        narrative.append(f"• Subsystem Health: All stages operating within normal tolerances (Healthy confidence: {cause_probs[0]:.1%}).")
        
    return "\n".join(narrative)

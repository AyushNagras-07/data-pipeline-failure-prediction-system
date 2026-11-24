"""
plot_generator.py
Generates publication-quality experimental evaluation charts for ANNA Course 2307411T (CO4).

Outputs 5 high-resolution figures to reports/figures/:
1. 1_architecture_comparison_metrics.png - Multi-metric bar chart comparing RNN vs LSTM vs GRU vs Attn-Models
2. 2_window_size_ablation.png            - Window size T in [3, 5, 7, 10] vs F1-Score and Latency
3. 3_training_convergence_curves.png     - Loss curves across 30 epochs showing Attention-GRU convergence
4. 4_roc_curves.png                     - ROC Curves with AUC for all evaluated models
5. 5_confusion_matrix_subsystems.png     - 5x5 Subsystem Root-Cause Classification Confusion Matrix
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Styling setup
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 200
})

def generate_all_plots(summary_json_path=None):
    if summary_json_path is None:
        summary_json_path = os.path.join(REPORTS_DIR, "ablation_summary.json")
        
    if not os.path.exists(summary_json_path):
        raise FileNotFoundError(f"Summary JSON not found at {summary_json_path}. Run hyperparameter_sweep.py first.")
        
    with open(summary_json_path, "r") as f:
        summary = json.load(f)
        
    saved_plots = []

    # -------------------------------------------------------------------------
    # Figure 1: Architecture Comparison Multi-Metric Bar Chart
    # -------------------------------------------------------------------------
    arch_data = summary["architectures"]
    df_arch = pd.DataFrame(arch_data)
    
    models = df_arch["architecture"].tolist()
    accuracies = [a * 100 for a in df_arch["accuracy"]]
    recalls = [r * 100 for r in df_arch["recall"]]
    f1_scores = [f * 100 for f in df_arch["f1_score"]]
    aucs = [u * 100 for u in df_arch["roc_auc"]]
    
    x = np.arange(len(models))
    width = 0.20
    
    fig, ax = plt.subplots(figsize=(10, 5.5))
    rects1 = ax.bar(x - 1.5*width, accuracies, width, label="Accuracy (%)", color="#1f77b4")
    rects2 = ax.bar(x - 0.5*width, recalls, width, label="Recall (%)", color="#ff7f0e")
    rects3 = ax.bar(x + 0.5*width, f1_scores, width, label="F1-Score (%)", color="#2ca02c")
    rects4 = ax.bar(x + 1.5*width, aucs, width, label="ROC-AUC (%)", color="#d62728")
    
    ax.set_ylabel("Performance Metric (%)")
    ax.set_title("Figure 1: Architectural Comparison on Telemetry Failure Prediction (ANNA Track 2)", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylim(0, 115)
    ax.legend(loc="upper left", ncol=4)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    
    # Label top bars
    for rects in [rects1, rects2, rects3, rects4]:
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(f"{height:.0f}",
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=7.5)
                            
    fig.tight_layout()
    p1 = os.path.join(FIGURES_DIR, "1_architecture_comparison_metrics.png")
    fig.savefig(p1, bbox_inches="tight")
    plt.close(fig)
    saved_plots.append(p1)

    # -------------------------------------------------------------------------
    # Figure 2: Window Size (Receptive Field) Ablation Study
    # -------------------------------------------------------------------------
    w_data = summary["window_ablation"]
    windows = sorted([int(k) for k in w_data.keys()])
    f1s = [w_data[str(w)]["f1_score"] * 100 for w in windows]
    stage_accs = [w_data[str(w)]["stage_acc"] * 100 for w in windows]
    latencies = [w_data[str(w)]["latency_ms"] for w in windows]
    
    fig, ax1 = plt.subplots(figsize=(9, 5))
    
    color1 = "#2ca02c"
    color2 = "#1f77b4"
    color3 = "#d62728"
    
    ax1.plot(windows, f1s, marker="o", linewidth=2.5, color=color1, label="Binary Risk F1-Score (%)")
    ax1.plot(windows, stage_accs, marker="s", linewidth=2.5, color=color2, label="Subsystem Stage Accuracy (%)")
    ax1.set_xlabel("Sliding Window Size T (Historical Executions Receptive Field)")
    ax1.set_ylabel("Accuracy / F1 Metrics (%)", color=color1)
    ax1.tick_params(axis="y")
    ax1.set_ylim(50, 105)
    
    ax2 = ax1.twinx()
    ax2.plot(windows, latencies, marker="^", linewidth=2.2, linestyle="--", color=color3, label="Inference Latency (ms)")
    ax2.set_ylabel("Single-Sample Inference Latency (ms)", color=color3)
    ax2.tick_params(axis="y", labelcolor=color3)
    ax2.grid(False)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")
    
    ax1.set_title("Figure 2: Impact of Temporal Window Size T on Accuracy vs. Latency (Attention-GRU)", pad=15)
    ax1.set_xticks(windows)
    fig.tight_layout()
    p2 = os.path.join(FIGURES_DIR, "2_window_size_ablation.png")
    fig.savefig(p2, bbox_inches="tight")
    plt.close(fig)
    saved_plots.append(p2)

    # -------------------------------------------------------------------------
    # Figure 3: Training & Validation Convergence Loss Curves
    # -------------------------------------------------------------------------
    histories = summary["arch_histories"]
    fig, (ax_train, ax_val) = plt.subplots(1, 2, figsize=(12, 4.8))
    
    colors = {
        "Vanilla RNN": "#7f7f7f",
        "Standard LSTM": "#ff7f0e",
        "Standard GRU": "#1f77b4",
        "Attention-LSTM": "#9467bd",
        "Attention-GRU (Ours)": "#2ca02c"
    }
    
    for name, hist in histories.items():
        c = colors.get(name, "black")
        lw = 2.5 if "Ours" in name else 1.6
        alpha = 1.0 if "Ours" in name else 0.75
        epochs_arr = np.arange(1, len(hist["train_loss"]) + 1)
        ax_train.plot(epochs_arr, hist["train_loss"], label=name, color=c, linewidth=lw, alpha=alpha)
        ax_val.plot(epochs_arr, hist["val_loss"], label=name, color=c, linewidth=lw, alpha=alpha)
        
    ax_train.set_title("Training Loss Convergence across Epochs")
    ax_train.set_xlabel("Epoch")
    ax_train.set_ylabel("Training Loss (BCE / Joint Multi-Task)")
    ax_train.grid(True, linestyle="--", alpha=0.6)
    
    ax_val.set_title("Validation Loss (Generalization Trajectory)")
    ax_val.set_xlabel("Epoch")
    ax_val.set_ylabel("Validation Loss")
    ax_val.grid(True, linestyle="--", alpha=0.6)
    ax_val.legend(loc="upper right", frameon=True)
    
    fig.suptitle("Figure 3: Neural Network Optimization Trajectory (Adam, lr=0.008)", y=1.02)
    fig.tight_layout()
    p3 = os.path.join(FIGURES_DIR, "3_training_convergence_curves.png")
    fig.savefig(p3, bbox_inches="tight")
    plt.close(fig)
    saved_plots.append(p3)

    # -------------------------------------------------------------------------
    # Figure 4: ROC Curves with Area Under Curve (AUC)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for name, hist in histories.items():
        y_true = np.array(hist["y_test"])
        y_probs = np.array(hist["probs"])
        if len(np.unique(y_true)) > 1:
            fpr, tpr, _ = roc_curve(y_true, y_probs)
            roc_auc = auc(fpr, tpr)
            c = colors.get(name, "black")
            lw = 2.5 if "Ours" in name else 1.8
            ax.plot(fpr, tpr, color=c, linewidth=lw, label=f"{name} (AUC = {roc_auc:.3f})")
            
    ax.plot([0, 1], [0, 1], color="navy", linestyle="--", linewidth=1.2, label="Random Guess (AUC = 0.500)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (Fall-out)")
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)")
    ax.set_title("Figure 4: Receiver Operating Characteristic (ROC) Comparative Analysis", pad=15)
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)
    
    fig.tight_layout()
    p4 = os.path.join(FIGURES_DIR, "4_roc_curves.png")
    fig.savefig(p4, bbox_inches="tight")
    plt.close(fig)
    saved_plots.append(p4)

    # -------------------------------------------------------------------------
    # Figure 5: Normalized Confusion Matrix for 5 Subsystems
    # -------------------------------------------------------------------------
    cm_data = summary.get("best_confusion_matrix")
    if cm_data:
        y_true_stage = np.array(cm_data["y_true"])
        y_pred_stage = np.array(cm_data["y_pred"])
        stage_names = summary["stage_names"]
        
        cm = confusion_matrix(y_true_stage, y_pred_stage, labels=range(len(stage_names)))
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)  # handle division by zero
        
        fig, ax = plt.subplots(figsize=(8, 6.5))
        cax = ax.matshow(cm_norm, cmap=plt.cm.Blues, vmin=0, vmax=1.0)
        cbar = fig.colorbar(cax)
        cbar.set_label("Classification Probability (Recall per Class)")
        
        ax.set_xticks(range(len(stage_names)))
        ax.set_yticks(range(len(stage_names)))
        ax.set_xticklabels(stage_names, rotation=25, ha="left")
        ax.set_yticklabels(stage_names)
        
        # Add text annotations inside cells
        for i in range(len(stage_names)):
            for j in range(len(stage_names)):
                raw_count = cm[i, j]
                pct = cm_norm[i, j]
                text_color = "white" if pct > 0.5 else "black"
                ax.text(j, i, f"{pct:.1%}\n(n={raw_count})",
                        ha="center", va="center", color=text_color, fontsize=9.5)
                        
        ax.set_xlabel("Predicted Pipeline Subsystem Component", labelpad=10)
        ax.set_ylabel("Actual Ground Truth Subsystem Component", labelpad=10)
        ax.set_title("Figure 5: Attention-GRU Subsystem Root Cause Confusion Matrix", pad=20)
        ax.grid(False)
        
        fig.tight_layout()
        p5 = os.path.join(FIGURES_DIR, "5_confusion_matrix_subsystems.png")
        fig.savefig(p5, bbox_inches="tight")
        plt.close(fig)
        saved_plots.append(p5)

    print(f"\n[INFO] Successfully generated {len(saved_plots)} publication figures in: {FIGURES_DIR}")
    for p in saved_plots:
        print(f"  • {os.path.basename(p)}")
    return saved_plots

if __name__ == "__main__":
    generate_all_plots()

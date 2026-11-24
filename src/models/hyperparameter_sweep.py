"""
hyperparameter_sweep.py
Automated Grid Search & Ablation Engine for Deep Sequential Models (ANNA Track 2: CO4).

Conducts rigorous experimental sweeps across:
1. Model Architectures: Vanilla RNN vs. LSTM vs. GRU vs. Attention-LSTM vs. Attention-GRU
2. Temporal Window Receptive Fields: T in [3, 5, 7, 10]
3. Hidden Unit Capacities: D_hidden in [12, 16, 24, 32, 64]
4. Optimizers: Adam vs. RMSprop vs. SGD (Momentum)

Logs: Train Loss, Val Loss, Test Loss, Accuracy, Precision, Recall, F1-Score,
      ROC-AUC, Subsystem Stage Accuracy, Inference Latency, and Parameter Count.
Outputs: reports/hyperparameter_sweep_results.csv & reports/ablation_summary.json
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(PROJECT_ROOT)

from src.features.sequence_generator import prepare_sequential_data
from src.models.rnn_model import PipelineRNN
from src.models.lstm_model import PipelineLSTM
from src.models.gru_model import PipelineGRU
from src.models.attention_model import AttentionGRU, AttentionLSTM

CSV_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "pipeline_telemetry.csv")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

def train_eval_model(model_cls, data, hidden_dim=24, lr=0.008, opt_name="Adam", epochs=30, is_dual_head=False):
    """
    Trains and evaluates a model architecture, recording epoch histories and final metrics.
    """
    X_train, y_train, y_stage_train = data["X_train"], data["y_train"], data["y_stage_train"]
    X_val, y_val, y_stage_val = data["X_val"], data["y_val"], data["y_stage_val"]
    X_test, y_test, y_stage_test = data["X_test"], data["y_test"], data["y_stage_test"]
    
    input_dim = X_train.shape[2]
    
    if is_dual_head:
        model = model_cls(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=5)
    else:
        model = model_cls(input_dim=input_dim, hidden_dim=hidden_dim)
        
    bce_loss_fn = nn.BCELoss()
    ce_loss_fn = nn.CrossEntropyLoss()
    
    if opt_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    elif opt_name == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=1e-4)
    elif opt_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
    train_loss_history = []
    val_loss_history = []
    
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        if is_dual_head:
            risk_preds, cause_logits, _ = model(X_train)
            loss = bce_loss_fn(risk_preds, y_train) + 0.5 * ce_loss_fn(cause_logits, y_stage_train)
        else:
            preds = model(X_train)
            loss = bce_loss_fn(preds, y_train)
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        
        train_loss_history.append(loss.item())
        
        # Validation loss
        model.eval()
        with torch.no_grad():
            if is_dual_head:
                v_risk, v_cause, _ = model(X_val)
                v_loss = bce_loss_fn(v_risk, y_val) + 0.5 * ce_loss_fn(v_cause, y_stage_val)
            else:
                v_preds = model(X_val)
                v_loss = bce_loss_fn(v_preds, y_val)
            val_loss_history.append(v_loss.item())
            
    train_time = time.time() - t0
    
    # Final Test Set Evaluation
    model.eval()
    with torch.no_grad():
        t_start_inf = time.time()
        for _ in range(50):
            if is_dual_head:
                _ = model(X_test[:1])
            else:
                _ = model(X_test[:1])
        latency_ms = ((time.time() - t_start_inf) / 50.0) * 1000.0
        
        if is_dual_head:
            test_risk, test_cause, test_attn = model(X_test)
            test_probs = test_risk.numpy()
            stage_preds = torch.argmax(test_cause, dim=1).numpy()
            stage_acc = float(accuracy_score(y_stage_test.numpy(), stage_preds))
        else:
            test_preds = model(X_test)
            test_probs = test_preds.numpy()
            stage_acc = None
            
        test_binary = (test_probs >= 0.5).astype(float)
        y_test_np = y_test.numpy()
        
        acc = float(accuracy_score(y_test_np, test_binary))
        rec = float(recall_score(y_test_np, test_binary, zero_division=0))
        prec = float(precision_score(y_test_np, test_binary, zero_division=0))
        f1 = float(f1_score(y_test_np, test_binary, zero_division=0))
        
        try:
            auc = float(roc_auc_score(y_test_np, test_probs))
        except ValueError:
            auc = 0.5
            
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "model": model,
        "train_loss_history": train_loss_history,
        "val_loss_history": val_loss_history,
        "final_val_loss": float(val_loss_history[-1]),
        "acc": acc,
        "rec": rec,
        "prec": prec,
        "f1": f1,
        "auc": auc,
        "stage_acc": stage_acc,
        "latency_ms": round(latency_ms, 3),
        "params": n_params,
        "train_time": round(train_time, 2),
        "test_probs": test_probs,
        "y_test": y_test_np,
        "stage_preds": stage_preds if is_dual_head else None,
        "y_stage_test": y_stage_test.numpy() if is_dual_head else None
    }

def run_comprehensive_sweep():
    print("=" * 80)
    print(" EXECUTING ANNA TRACK 2 HYPERPARAMETER SWEEP & ABLATION STUDY (CO4)")
    print("=" * 80)
    
    results = []
    
    # -------------------------------------------------------------------------
    # 1. Architecture Comparison (T = 3, Hidden = 24, Adam)
    # -------------------------------------------------------------------------
    print("\n[Phase 1/4] Model Architecture Comparative Evaluation (T=3, D=24)...")
    data_t3 = prepare_sequential_data(CSV_PATH, window_size=3)
    
    arch_configs = [
        ("Vanilla RNN", PipelineRNN, False),
        ("Standard LSTM", PipelineLSTM, False),
        ("Standard GRU", PipelineGRU, False),
        ("Attention-LSTM", AttentionLSTM, True),
        ("Attention-GRU (Ours)", AttentionGRU, True)
    ]
    
    arch_histories = {}
    best_attention_gru_res = None
    
    for name, cls, is_dual in arch_configs:
        res = train_eval_model(cls, data_t3, hidden_dim=24, is_dual_head=is_dual, epochs=30)
        arch_histories[name] = {
            "train_loss": res["train_loss_history"],
            "val_loss": res["val_loss_history"],
            "probs": res["test_probs"].tolist(),
            "y_test": res["y_test"].tolist()
        }
        if name == "Attention-GRU (Ours)":
            best_attention_gru_res = res
            
        results.append({
            "experiment": "Architecture_Comparison",
            "architecture": name,
            "window_size": 3,
            "hidden_dim": 24,
            "optimizer": "Adam",
            "learning_rate": 0.008,
            "val_loss": res["final_val_loss"],
            "accuracy": res["acc"],
            "recall": res["rec"],
            "precision": res["prec"],
            "f1_score": res["f1"],
            "roc_auc": res["auc"],
            "stage_accuracy": res["stage_acc"] if res["stage_acc"] is not None else -1.0,
            "latency_ms": res["latency_ms"],
            "parameters": res["params"],
            "train_time_s": res["train_time"]
        })
        print(f" • {name:<22} | Acc: {res['acc']:.2%} | Rec: {res['rec']:.2%} | F1: {res['f1']:.2%} | Latency: {res['latency_ms']}ms")

    # -------------------------------------------------------------------------
    # 2. Window Size Ablation Study: T in [3, 5, 7, 10]
    # -------------------------------------------------------------------------
    print("\n[Phase 2/4] Window Size Receptive Field Ablation (Attention-GRU)...")
    window_sizes = [3, 5, 7, 10]
    window_results = {}
    
    for w in window_sizes:
        data_w = prepare_sequential_data(CSV_PATH, window_size=w)
        res_w = train_eval_model(AttentionGRU, data_w, hidden_dim=24, is_dual_head=True, epochs=30)
        window_results[w] = {
            "accuracy": res_w["acc"],
            "f1_score": res_w["f1"],
            "recall": res_w["rec"],
            "stage_acc": res_w["stage_acc"],
            "latency_ms": res_w["latency_ms"],
            "total_sequences": len(data_w["X_train"]) + len(data_w["X_val"]) + len(data_w["X_test"])
        }
        results.append({
            "experiment": "Window_Size_Ablation",
            "architecture": "Attention-GRU",
            "window_size": w,
            "hidden_dim": 24,
            "optimizer": "Adam",
            "learning_rate": 0.008,
            "val_loss": res_w["final_val_loss"],
            "accuracy": res_w["acc"],
            "recall": res_w["rec"],
            "precision": res_w["prec"],
            "f1_score": res_w["f1"],
            "roc_auc": res_w["auc"],
            "stage_accuracy": res_w["stage_acc"],
            "latency_ms": res_w["latency_ms"],
            "parameters": res_w["params"],
            "train_time_s": res_w["train_time"]
        })
        print(f" • Window T={w:<2} | F1: {res_w['f1']:.2%} | Stage Acc: {res_w['stage_acc']:.2%} | Latency: {res_w['latency_ms']}ms")

    # -------------------------------------------------------------------------
    # 3. Hidden Capacity Ablation: D_hidden in [12, 16, 24, 32, 64]
    # -------------------------------------------------------------------------
    print("\n[Phase 3/4] Hidden State Capacity Ablation (Attention-GRU, T=3)...")
    hidden_dims = [12, 16, 24, 32, 64]
    hidden_results = {}
    
    for h in hidden_dims:
        res_h = train_eval_model(AttentionGRU, data_t3, hidden_dim=h, is_dual_head=True, epochs=25)
        hidden_results[h] = {
            "val_loss": res_h["final_val_loss"],
            "f1_score": res_h["f1"],
            "params": res_h["params"],
            "latency_ms": res_h["latency_ms"]
        }
        results.append({
            "experiment": "Hidden_Dimension_Ablation",
            "architecture": "Attention-GRU",
            "window_size": 3,
            "hidden_dim": h,
            "optimizer": "Adam",
            "learning_rate": 0.008,
            "val_loss": res_h["final_val_loss"],
            "accuracy": res_h["acc"],
            "recall": res_h["rec"],
            "precision": res_h["prec"],
            "f1_score": res_h["f1"],
            "roc_auc": res_h["auc"],
            "stage_accuracy": res_h["stage_acc"],
            "latency_ms": res_h["latency_ms"],
            "parameters": res_h["params"],
            "train_time_s": res_h["train_time"]
        })
        print(f" • Hidden Dim D={h:<2} | Params: {res_h['params']:<5} | F1: {res_h['f1']:.2%} | Loss: {res_h['final_val_loss']:.4f}")

    # -------------------------------------------------------------------------
    # 4. Optimizer Sensitivity: Adam vs. RMSprop vs. SGD
    # -------------------------------------------------------------------------
    print("\n[Phase 4/4] Optimizer Convergence Sensitivity (Attention-GRU, T=3, D=24)...")
    optimizers = ["Adam", "RMSprop", "SGD"]
    for opt in optimizers:
        res_opt = train_eval_model(AttentionGRU, data_t3, hidden_dim=24, opt_name=opt, is_dual_head=True, epochs=25)
        results.append({
            "experiment": "Optimizer_Sensitivity",
            "architecture": "Attention-GRU",
            "window_size": 3,
            "hidden_dim": 24,
            "optimizer": opt,
            "learning_rate": 0.008,
            "val_loss": res_opt["final_val_loss"],
            "accuracy": res_opt["acc"],
            "recall": res_opt["rec"],
            "precision": res_opt["prec"],
            "f1_score": res_opt["f1"],
            "roc_auc": res_opt["auc"],
            "stage_accuracy": res_opt["stage_acc"],
            "latency_ms": res_opt["latency_ms"],
            "parameters": res_opt["params"],
            "train_time_s": res_opt["train_time"]
        })
        print(f" • Optimizer: {opt:<8} | F1: {res_opt['f1']:.2%} | Final Val Loss: {res_opt['final_val_loss']:.4f}")

    # Save to CSV
    df_results = pd.DataFrame(results)
    csv_out = os.path.join(REPORTS_DIR, "hyperparameter_sweep_results.csv")
    df_results.to_csv(csv_out, index=False)
    print(f"\n[INFO] Comprehensive grid search logged to: {csv_out}")
    
    # Save structured summary JSON for figures and report
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_experiments": len(results),
        "architectures": [r for r in results if r["experiment"] == "Architecture_Comparison"],
        "window_ablation": window_results,
        "hidden_ablation": hidden_results,
        "arch_histories": arch_histories,
        "best_confusion_matrix": {
            "y_true": best_attention_gru_res["y_stage_test"].tolist(),
            "y_pred": best_attention_gru_res["stage_preds"].tolist()
        } if best_attention_gru_res else None,
        "stage_names": data_t3["stage_names"]
    }
    json_out = os.path.join(REPORTS_DIR, "ablation_summary.json")
    with open(json_out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[INFO] Structured experimental summary saved to: {json_out}")
    
    return df_results, summary

if __name__ == "__main__":
    run_comprehensive_sweep()

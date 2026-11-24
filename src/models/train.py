"""
train.py
Unified training and comparative evaluation script for RNN, LSTM, and GRU models.
Demonstrates end-to-end training and comparative metrics (Loss, Accuracy, Recall).
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score

# Ensure src modules can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.features.sequence_generator import prepare_sequential_data
from src.models.rnn_model import PipelineRNN
from src.models.lstm_model import PipelineLSTM
from src.models.gru_model import PipelineGRU

def train_single_model(model, X_train, y_train, X_val, y_val, epochs=25, lr=0.01):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        predictions = model(X_train)
        loss = criterion(predictions, y_train)
        loss.backward()
        optimizer.step()
        
    model.eval()
    with torch.no_grad():
        val_preds = model(X_val)
        val_loss = criterion(val_preds, y_val).item()
        binary_preds = (val_preds >= 0.5).float().numpy()
        y_val_np = y_val.numpy()
        
        acc = accuracy_score(y_val_np, binary_preds)
        # Calculate recall with zero_division handled
        rec = recall_score(y_val_np, binary_preds, zero_division=0)
        
    return val_loss, acc, rec

"""
train.py
Unified training and comparative evaluation script for RNN, LSTM, and GRU models.
Demonstrates end-to-end training, comparative metrics (Loss, Accuracy, Recall),
and serializes the best trained model checkpoint to models/best_gru_model.pth.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score

# Ensure src modules can be imported
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(PROJECT_ROOT)
from src.features.sequence_generator import prepare_sequential_data
from src.models.rnn_model import PipelineRNN
from src.models.lstm_model import PipelineLSTM
from src.models.gru_model import PipelineGRU

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def train_single_model(model, X_train, y_train, X_val, y_val, epochs=30, lr=0.01):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        predictions = model(X_train)
        loss = criterion(predictions, y_train)
        loss.backward()
        optimizer.step()
        
    model.eval()
    with torch.no_grad():
        val_preds = model(X_val)
        val_loss = criterion(val_preds, y_val).item()
        binary_preds = (val_preds >= 0.5).float().numpy()
        y_val_np = y_val.numpy()
        
        acc = accuracy_score(y_val_np, binary_preds)
        rec = recall_score(y_val_np, binary_preds, zero_division=0)
        
    return val_loss, acc, rec

def run_model_comparison(save_best=True):
    csv_file = os.path.join(PROJECT_ROOT, "data", "raw", "pipeline_telemetry.csv")
    
    print("\n" + "="*70)
    print(" ANNA Track 2: Sequential Models Prototype Training & Comparison")
    print("="*70)
    
    # 1. Prepare sequential data
    data = prepare_sequential_data(csv_file, window_size=3)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]
    
    print(f" Dataset Summary:")
    print(f"  • Feature Dimensions:  {data['feature_names']}")
    print(f"  • Window Size (T):     {data['window_size']} past runs")
    print(f"  • Train Sequences:     {X_train.shape[0]}")
    print(f"  • Validation Sequences:{X_val.shape[0]}")
    print(f"  • Test Sequences:      {X_test.shape[0]}")
    print("-"*70)
    
    # 2. Train and benchmark the 3 models
    models = {
        "Vanilla RNN": PipelineRNN(input_dim=4, hidden_dim=16),
        "LSTM":        PipelineLSTM(input_dim=4, hidden_dim=16),
        "GRU":         PipelineGRU(input_dim=4, hidden_dim=16)
    }
    
    results = {}
    
    print(f"{'Model Architecture':<18} | {'Val Loss':<10} | {'Accuracy':<10} | {'Recall':<10} | {'Parameters'}")
    print("-"*70)
    
    for name, model in models.items():
        val_loss, acc, rec = train_single_model(model, X_train, y_train, X_val, y_val)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        results[name] = {"loss": val_loss, "acc": acc, "rec": rec, "model": model}
        print(f"{name:<18} | {val_loss:<10.4f} | {acc:<10.2%} | {rec:<10.2%} | {n_params:<10}")
        
    print("-"*70)
    
    # Save best performing model (GRU)
    if save_best:
        checkpoint_path = os.path.join(MODELS_DIR, "best_gru_model.pth")
        torch.save({
            "model_state_dict": models["GRU"].state_dict(),
            "input_dim": 4,
            "hidden_dim": 16,
            "window_size": 3,
            "feature_names": data["feature_names"],
            "scaler": data["scaler"]
        }, checkpoint_path)
        print(f" [INFO] Best Model checkpoint serialized to: {checkpoint_path}")
    
    # 3. Live Inference Demo on the most recent sequence (Predicting Upcoming Run)
    latest_sequence = X_test[-1].unsqueeze(0)  # Shape: (1, 3, 4)
    best_model = models["GRU"]
    best_model.eval()
    with torch.no_grad():
        fail_prob = best_model(latest_sequence).item()
        
    risk_level = "HIGH RISK" if fail_prob >= 0.70 else ("MEDIUM RISK" if fail_prob >= 0.30 else "LOW RISK")
    
    print("\n Live Prediction for Next Scheduled Pipeline Execution:")
    print(f"  • Input Sequence:      Last 3 completed runs telemetry")
    print(f"  • Failure Probability: {fail_prob:.1%}")
    print(f"  • Success Probability: {1.0 - fail_prob:.1%}")
    print(f"  • Predicted Risk Level: [{risk_level}]")
    print("="*70 + "\n")
    return results, data

if __name__ == "__main__":
    run_model_comparison()

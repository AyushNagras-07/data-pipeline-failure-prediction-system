"""
attention_model.py
Advanced Sequential Neural Network Architectures with Temporal Attention (ANNA Track 2: CO3).

Implements:
1. TemporalAttention: Bahdanau-style additive attention over sequential telemetry time-steps.
2. AttentionGRU: Dual-head GRU with temporal attention mechanism:
   - Head 1 (Binary Risk Head): P(Failure) in [0, 1]
   - Head 2 (Root Cause Head): Softmax distribution over 5 pipeline component stages
   - Interpretability: Exposes attention alignment weights (alpha_t) across past T runs.
3. AttentionLSTM: Dual-head LSTM with temporal attention for architecture ablation comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalAttention(nn.Module):
    """
    Additive Temporal Attention Mechanism.
    Computes importance alignment score for each time-step in a historical window [t-W, ..., t-1]:
      e_t = v^T tanh(W_h * h_t + b)
      alpha_t = softmax(e_t)
      c = sum(alpha_t * h_t)
    """
    def __init__(self, hidden_dim, attention_dim=16):
        super(TemporalAttention, self).__init__()
        self.attn_projection = nn.Linear(hidden_dim, attention_dim)
        self.v = nn.Linear(attention_dim, 1, bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, recurrent_outputs):
        # recurrent_outputs shape: (batch_size, seq_len, hidden_dim)
        # 1. Project to attention latent space: (batch_size, seq_len, attention_dim)
        energy = self.tanh(self.attn_projection(recurrent_outputs))
        
        # 2. Compute scalar score per time-step: (batch_size, seq_len, 1)
        scores = self.v(energy)
        
        # 3. Softmax normalization across time window: (batch_size, seq_len, 1)
        weights = self.softmax(scores)
        
        # 4. Context vector as weighted linear combination: (batch_size, hidden_dim)
        context = torch.sum(recurrent_outputs * weights, dim=1)
        
        # Return context vector and squeezed 1D attention weights per sequence: (batch_size, seq_len)
        return context, weights.squeeze(-1)

class AttentionGRU(nn.Module):
    """
    Dual-Head Temporal Attention GRU Network.
    - Recurrent Backbone: Gated Recurrent Unit capturing temporal degradation trajectories.
    - Attention Engine: Weights relative contribution of each historical run.
    - Fusion Layer: Combines global temporal context with latest run state.
    - Head 1: Binary Failure Risk Probability.
    - Head 2: Multi-Class Pipeline Stage Root Cause Logits.
    """
    def __init__(self, input_dim=4, hidden_dim=24, attention_dim=16, num_classes=5, num_layers=1, dropout=0.1):
        super(AttentionGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.attention = TemporalAttention(hidden_dim=hidden_dim, attention_dim=attention_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Fusion: concatenate context vector (hidden_dim) and last step hidden state (hidden_dim)
        self.fusion_fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
        
        # Head 1: Binary Risk Head (Sigmoid probability)
        self.risk_head = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
        # Head 2: Root Cause Stage Classification Head (Logits for Cross-Entropy)
        self.cause_head = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        gru_out, _ = self.gru(x)  # (batch_size, seq_len, hidden_dim)
        
        # Compute temporal attention context and weights
        context, attn_weights = self.attention(gru_out)  # context: (batch, hidden_dim), attn_weights: (batch, seq_len)
        
        # Last time-step hidden state
        last_step = gru_out[:, -1, :]  # (batch, hidden_dim)
        
        # Fuse context + latest state
        fused = torch.cat([context, last_step], dim=1)  # (batch, hidden_dim * 2)
        fused = self.dropout(self.relu(self.fusion_fc(fused)))  # (batch, hidden_dim)
        
        # Head 1: Risk Probability
        risk_logit = self.risk_head(fused)
        risk_prob = self.sigmoid(risk_logit).squeeze(-1)  # (batch,)
        
        # Head 2: Multi-Class Root Cause Stage Logits
        cause_logits = self.cause_head(fused)  # (batch, num_classes)
        
        return risk_prob, cause_logits, attn_weights

class AttentionLSTM(nn.Module):
    """
    Dual-Head Temporal Attention LSTM Network for architectural ablation study.
    Replaces GRU cell with LSTM cell (with separate memory cell c_t and hidden state h_t).
    """
    def __init__(self, input_dim=4, hidden_dim=24, attention_dim=16, num_classes=5, num_layers=1, dropout=0.1):
        super(AttentionLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.attention = TemporalAttention(hidden_dim=hidden_dim, attention_dim=attention_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.fusion_fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
        
        # Dual classification heads
        self.risk_head = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.cause_head = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, hidden_dim)
        context, attn_weights = self.attention(lstm_out)
        last_step = lstm_out[:, -1, :]
        
        fused = torch.cat([context, last_step], dim=1)
        fused = self.dropout(self.relu(self.fusion_fc(fused)))
        
        risk_logit = self.risk_head(fused)
        risk_prob = self.sigmoid(risk_logit).squeeze(-1)
        cause_logits = self.cause_head(fused)
        
        return risk_prob, cause_logits, attn_weights

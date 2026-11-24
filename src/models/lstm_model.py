"""
lstm_model.py
Long Short-Term Memory (LSTM) network with gating mechanism to preserve long-range telemetry dependencies.
"""

import torch
import torch.nn as nn

class PipelineLSTM(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=16, num_layers=1, dropout=0.1):
        super(PipelineLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        out, (hn, cn) = self.lstm(x)
        # Take the output of the last time step
        last_step_out = out[:, -1, :]
        last_step_out = self.dropout(last_step_out)
        logits = self.fc(last_step_out)
        prob = self.sigmoid(logits)
        return prob.squeeze(1)

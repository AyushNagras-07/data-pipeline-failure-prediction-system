"""
gru_model.py
Gated Recurrent Unit (GRU) network providing parameter-efficient gating for telemetry failure prediction.
"""

import torch
import torch.nn as nn

class PipelineGRU(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=16, num_layers=1, dropout=0.1):
        super(PipelineGRU, self).__init__()
        self.gru = nn.GRU(
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
        out, hn = self.gru(x)
        # Take the output of the last time step
        last_step_out = out[:, -1, :]
        last_step_out = self.dropout(last_step_out)
        logits = self.fc(last_step_out)
        prob = self.sigmoid(logits)
        return prob.squeeze(1)

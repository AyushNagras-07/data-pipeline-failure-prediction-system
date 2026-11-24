"""
rnn_model.py
Vanilla Recurrent Neural Network (RNN) baseline for pipeline failure prediction.
"""

import torch
import torch.nn as nn

class PipelineRNN(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=16, num_layers=1, dropout=0.1):
        super(PipelineRNN, self).__init__()
        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity='tanh'
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        out, _ = self.rnn(x)
        # Take the output of the last time step
        last_step_out = out[:, -1, :]
        last_step_out = self.dropout(last_step_out)
        logits = self.fc(last_step_out)
        prob = self.sigmoid(logits)
        return prob.squeeze(1)

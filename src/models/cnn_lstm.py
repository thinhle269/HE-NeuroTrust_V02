"""Compact 1D-CNN-LSTM for continuous-authentication sensor windows.

Ported (in compact form) from the V-TrustFL architecture, this model
consumes HMOG-style sensor windows of shape ``(T, C)`` - ``T`` timesteps of
``C`` inertial channels (accelerometer + gyroscope) - and outputs a
genuine/impostor class score.

The network is deliberately small (a few thousand parameters) for the same
reason as the MLP in :mod:`src.models.mlp`: Paillier homomorphic
aggregation encrypts one ciphertext per parameter, so the whole FL + HE
pipeline stays tractable only if the model is compact.  A 1D convolutional
front-end extracts local inertial patterns; a small LSTM captures the
temporal dynamics of the window; a linear head produces the two-class
authentication logits.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class CNNLSTMAuth(nn.Module):
    def __init__(self, input_shape: Sequence[int], num_classes: int = 2,
                 conv_channels: int = 8, lstm_hidden: int = 16,
                 dropout: float = 0.2):
        super().__init__()
        self.T, self.C = int(input_shape[0]), int(input_shape[1])
        self.features = nn.Sequential(
            nn.Conv1d(self.C, conv_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.lstm = nn.LSTM(conv_channels, lstm_hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(lstm_hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.view(x.size(0), self.T, self.C)
        x = x.transpose(1, 2)
        z = self.features(x)                 # (B, conv_channels, T')
        z = z.transpose(1, 2)                # (B, T', conv_channels)
        out, (h, _) = self.lstm(z)           # h: (1, B, lstm_hidden)
        z = self.dropout(h[-1])              # (B, lstm_hidden)
        return self.head(z)                  # (B, num_classes)

    def count_parameters(self, trainable_only: bool = True) -> int:
        ps = (p for p in self.parameters() if (p.requires_grad or not trainable_only))
        return sum(p.numel() for p in ps)

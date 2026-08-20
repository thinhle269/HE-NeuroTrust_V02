"""Compact MLP for tabular IDS.

We deliberately use a *small* MLP (a few thousand parameters) for two reasons:

1.  The CIC-IoT-2023 features are already aggregate statistics per flow, so
    extra depth seldom helps and tends to overfit.
2.  Pairing the model with Paillier homomorphic aggregation - which encrypts
    one ciphertext per scalar weight - is computationally heavy.  Keeping the
    parameter count small makes the HE pipeline tractable on a single
    workstation, while still leaving the architecture realistic for an IDS
    paper.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


_ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "leaky_relu": nn.LeakyReLU,
}


class MLPIDS(nn.Module):
    def __init__(self, in_features: int, num_classes: int,
                 hidden_sizes: List[int] = (64, 32), dropout: float = 0.2,
                 activation: str = "relu"):
        super().__init__()
        if activation not in _ACTIVATIONS:
            raise ValueError(f"Unknown activation: {activation}")
        act_cls = _ACTIVATIONS[activation]
        layers: List[nn.Module] = []
        prev = in_features
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(act_cls())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def count_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


def build_model(cfg, in_features: int, num_classes: int):
    """Build the model for the configured ``model.type``.

    ``mlp`` (default) uses ``in_features``; ``cnn_lstm`` uses
    ``model.input_shape`` = (T, C) for sequence inputs (HMOG).
    """
    model_cfg = cfg.model
    mtype = str(model_cfg.get("type", "mlp")) if hasattr(model_cfg, "get") else "mlp"
    if mtype == "cnn_lstm":
        from .cnn_lstm import CNNLSTMAuth
        return CNNLSTMAuth(
            input_shape=list(model_cfg.input_shape),
            num_classes=num_classes,
            conv_channels=int(model_cfg.get("conv_channels", 8)),
            lstm_hidden=int(model_cfg.get("lstm_hidden", 16)),
            dropout=float(model_cfg.dropout),
        )
    return MLPIDS(
        in_features=in_features,
        num_classes=num_classes,
        hidden_sizes=list(model_cfg.hidden_sizes),
        dropout=float(model_cfg.dropout),
        activation=str(model_cfg.activation),
    )

"""PyTorch ``Dataset``/``DataLoader`` helpers for in-memory tabular data."""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class TabularDataset(Dataset):
    """Lightweight wrapper around two NumPy arrays already on CPU.

    We pre-convert to ``torch.Tensor`` once so that the per-batch overhead
    becomes a slice; on most networks the bottleneck then shifts to the
    forward pass rather than to ``__getitem__``.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X/y length mismatch: {X.shape[0]} vs {y.shape[0]}")
        self.X = torch.from_numpy(np.ascontiguousarray(X)).float()
        self.y = torch.from_numpy(np.ascontiguousarray(y)).long()

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def build_loaders(X_train: np.ndarray, y_train: np.ndarray,
                  X_val: np.ndarray, y_val: np.ndarray,
                  X_test: np.ndarray, y_test: np.ndarray,
                  batch_size: int = 256,
                  num_workers: int = 0,
                  pin_memory: Optional[bool] = None):
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    train = DataLoader(TabularDataset(X_train, y_train), batch_size=batch_size,
                       shuffle=True, num_workers=num_workers, pin_memory=pin_memory,
                       drop_last=False)
    val = DataLoader(TabularDataset(X_val, y_val), batch_size=batch_size * 4,
                     shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test = DataLoader(TabularDataset(X_test, y_test), batch_size=batch_size * 4,
                      shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    return train, val, test

"""Helper functions for flattening / unflattening model parameters.

Federated aggregation - whether in plaintext or under encryption - benefits
from working with a *single* flat vector per client instead of a nested
``state_dict``.  These helpers preserve parameter shapes so the aggregated
vector can be unpacked back into the model.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch


def flatten_state(state: Dict[str, torch.Tensor]) -> Tuple[np.ndarray, List[Tuple[str, Tuple[int, ...]]]]:
    """Concatenate every parameter tensor into a single ``np.float32`` vector.

    Returns the flat vector and a schema (list of ``(name, shape)``) that
    :func:`unflatten_state` needs to invert the operation.
    """
    chunks = []
    schema: List[Tuple[str, Tuple[int, ...]]] = []
    for name, tensor in state.items():
        arr = tensor.detach().cpu().numpy().astype(np.float32)
        chunks.append(arr.reshape(-1))
        schema.append((name, tuple(tensor.shape)))
    return np.concatenate(chunks, axis=0), schema


def unflatten_state(vec: np.ndarray,
                    schema: Sequence[Tuple[str, Tuple[int, ...]]]) -> "OrderedDict[str, torch.Tensor]":
    out: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    offset = 0
    for name, shape in schema:
        n = int(np.prod(shape)) if shape else 1
        slc = vec[offset:offset + n]
        out[name] = torch.from_numpy(np.asarray(slc, dtype=np.float32).reshape(shape).copy())
        offset += n
    if offset != vec.size:
        raise ValueError(f"Schema covers {offset} elements but vector has {vec.size}")
    return out


def plaintext_aggregate(updates: Sequence[np.ndarray],
                        weights: Sequence[float]) -> np.ndarray:
    """Plaintext weighted average of flat update vectors.

    Used as the "no-HE" baseline and as a sanity check that the encrypted
    aggregation produces the same numerical result (up to quantisation
    error).
    """
    if not updates:
        raise ValueError("No updates supplied")
    if len(updates) != len(weights):
        raise ValueError("updates / weights length mismatch")
    total_w = float(sum(weights))
    if total_w <= 0:
        raise ValueError("Total weight must be positive")
    acc = np.zeros_like(updates[0], dtype=np.float64)
    for u, w in zip(updates, weights):
        acc += float(w) * u.astype(np.float64)
    acc /= total_w
    return acc.astype(np.float32)

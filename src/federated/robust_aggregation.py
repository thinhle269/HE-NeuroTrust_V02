"""Byzantine-robust aggregation baselines used for comparison in the paper.

We re-implement four standard methods from the FL literature so the paper can
report a meaningful comparison alongside the proposed full system:

* **FedMedian** (Yin et al., ICML 2018):  the aggregated update at
  coordinate *j* is the median of all clients' values at coordinate *j*.
  Tolerates up to ``n/2`` Byzantine clients without strong assumptions.

* **Krum / Multi-Krum** (Blanchard et al., NeurIPS 2017):  for each candidate
  client *i* compute the sum of squared distances to the *k = n - f - 2*
  closest other clients (``f`` = #assumed Byzantine).  Pick the client with
  the smallest score; Multi-Krum averages the top *m* such clients.

* **TrimmedMean** (Yin et al., ICML 2018):  for each coordinate drop the
  ``beta`` smallest and ``beta`` largest values and average the rest.
  Tolerates up to ``beta/n`` fraction of Byzantine clients.

* **FedProx** (Li et al., MLSys 2020):  *not* an aggregation method but a
  client-side modification - we keep the FedAvg server-side recipe and only
  flag the proximal coefficient ``mu`` which the client will apply during
  local training.  Implemented in :mod:`client` for that reason.

All methods operate on **plaintext** flat-update vectors.  They are
incompatible with Paillier ciphertexts (Krum needs pairwise distances,
median / trimmed-mean need sorting) - we surface this in the paper as the
fundamental privacy-vs-robustness trade-off that the proposed system breaks.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def fedavg(updates: Sequence[np.ndarray], weights: Sequence[float]) -> np.ndarray:
    """Standard weighted average (kept here for symmetric dispatch)."""
    if not updates:
        raise ValueError("No updates supplied")
    if len(updates) != len(weights):
        raise ValueError("updates / weights length mismatch")
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("Total weight must be positive")
    stacked = np.stack([u.astype(np.float64) for u in updates], axis=0)
    w = np.asarray(weights, dtype=np.float64).reshape(-1, 1)
    return ((w * stacked).sum(axis=0) / total).astype(np.float32)


def fedmedian(updates: Sequence[np.ndarray], weights: Sequence[float] = None) -> np.ndarray:
    """Coordinate-wise median across clients.

    ``weights`` is accepted for a uniform call signature but is *ignored* -
    Yin et al.'s analysis assumes equal-weighted contributions.
    """
    if not updates:
        raise ValueError("No updates supplied")
    stacked = np.stack([u.astype(np.float64) for u in updates], axis=0)
    return np.median(stacked, axis=0).astype(np.float32)


def trimmed_mean(updates: Sequence[np.ndarray],
                 weights: Sequence[float] = None,
                 trim_ratio: float = 0.2) -> np.ndarray:
    """Coordinate-wise trimmed mean.

    Drops the ``beta = floor(trim_ratio * n)`` smallest *and* largest values
    per coordinate before averaging.  When ``2 * beta >= n`` we fall back to
    the coordinate-wise median.
    """
    if not updates:
        raise ValueError("No updates supplied")
    n = len(updates)
    beta = int(np.floor(float(trim_ratio) * n))
    if 2 * beta >= n:
        return fedmedian(updates)
    stacked = np.stack([u.astype(np.float64) for u in updates], axis=0)
    sorted_arr = np.sort(stacked, axis=0)
    trimmed = sorted_arr[beta: n - beta]
    return trimmed.mean(axis=0).astype(np.float32)


def _krum_scores(updates: Sequence[np.ndarray], num_byzantine: int) -> np.ndarray:
    """Krum closeness score: sum of distances to k nearest neighbours."""
    n = len(updates)
    f = max(0, int(num_byzantine))
    k = n - f - 2
    if k < 1:
        import warnings
        warnings.warn(
            f"Krum is undefined for n={n}, f={f} (requires n > 2f+2); "
            f"the aggregate in this regime is not meaningful.", RuntimeWarning)
        k = 1
    flat = np.stack([u.astype(np.float64).reshape(-1) for u in updates], axis=0)
    sq_norms = np.sum(flat * flat, axis=1)
    pairwise = sq_norms[:, None] + sq_norms[None, :] - 2.0 * (flat @ flat.T)
    np.fill_diagonal(pairwise, np.inf)            # ignore self in nearest search
    sorted_d = np.sort(pairwise, axis=1)
    scores = sorted_d[:, :k].sum(axis=1)
    return scores


def krum(updates: Sequence[np.ndarray], weights: Sequence[float] = None,
         num_byzantine: int = 1) -> np.ndarray:
    """Single-Krum: return the update from the client with the lowest score."""
    if not updates:
        raise ValueError("No updates supplied")
    if len(updates) == 1:
        return updates[0].astype(np.float32)
    scores = _krum_scores(updates, num_byzantine)
    chosen = int(np.argmin(scores))
    return updates[chosen].astype(np.float32)


def multi_krum(updates: Sequence[np.ndarray], weights: Sequence[float] = None,
               num_byzantine: int = 1, m: int = None) -> np.ndarray:
    """Multi-Krum: average the *m* updates with the smallest Krum scores."""
    if not updates:
        raise ValueError("No updates supplied")
    n = len(updates)
    if n == 1:
        return updates[0].astype(np.float32)
    if m is None:
        m = max(1, n - max(0, int(num_byzantine)))
    m = max(1, min(int(m), n))
    scores = _krum_scores(updates, num_byzantine)
    top = np.argsort(scores)[:m]
    selected = [updates[i] for i in top]
    return fedavg(selected, [1.0] * m)


def bulyan(updates: Sequence[np.ndarray], weights: Sequence[float] = None,
           num_byzantine: int = 1, beta: int = None) -> np.ndarray:
    """Bulyan (El Mhamdi et al., ICML 2018): two-stage Byzantine-robust agg.

    1. Multi-Krum selects ``m = n - 2f`` updates that are mutually close
       (lowest Krum scores).
    2. Coordinate-wise trimmed mean over those m updates with trim parameter
       ``beta = f`` (drop the f-smallest and f-largest values per coordinate
       inside the *selected* set).

    Tolerates up to ``(n - 3) / 4`` Byzantine clients with theoretical
    guarantees stronger than either Krum or Trimmed-Mean alone.
    """
    if not updates:
        raise ValueError("No updates supplied")
    n = len(updates)
    f = max(0, int(num_byzantine))
    if n == 1:
        return updates[0].astype(np.float32)
    f = min(f, max(0, (n - 3) // 4))
    m = max(1, n - 2 * f)
    scores = _krum_scores(updates, f)
    selected_idx = np.argsort(scores)[:m]
    selected = [updates[i] for i in selected_idx]
    if beta is None:
        beta = min(f, max(0, (m - 1) // 2))
    return trimmed_mean(selected, trim_ratio=beta / max(m, 1))


def foolsgold(updates: Sequence[np.ndarray],
              weights: Sequence[float] = None,
              history: Sequence[np.ndarray] = None,
              kappa: float = 1.0,
              eps: float = 1e-5) -> np.ndarray:
    """FoolsGold (Fung et al., RAID 2020): cosine-similarity re-weighting.

    The original algorithm uses the *historical* cumulative update of each
    client; for round-1 (or when the caller does not provide it) we fall
    back to the current-round updates only.  This is the documented
    "single-round" variant used by several follow-up FL benchmarks.

    Algorithm (single-round form):

        For each pair (i, j) with i != j:   cs_ij = cos(u_i, u_j)
        For each client i:                  cs_i^max = max_{j != i} cs_ij
        Pardon:                             For each i, if cs_i^max <
                                            max_j cs_j^max, scale cs_i^max
                                            down so honest "outliers" are
                                            not penalised for being unique.
        Weight:                             w_i = 1 - cs_i^max  (then clip
                                            to [0, 1] and renormalise)
        Logit confidence boost:             w_i = kappa * logit(w_i)
                                            then sigmoid back -> emphasises
                                            extremes.

    Returns the FoolsGold-weighted mean of ``updates``.
    """
    if not updates:
        raise ValueError("No updates supplied")
    n = len(updates)
    if n == 1:
        return updates[0].astype(np.float32)
    src = history if history is not None and len(history) == n else updates
    flat = np.stack([u.astype(np.float64).reshape(-1) for u in src], axis=0)
    norms = np.linalg.norm(flat, axis=1, keepdims=True) + eps
    normed = flat / norms
    cs = normed @ normed.T
    np.fill_diagonal(cs, 0.0)                      # ignore self-similarity
    v = cs.max(axis=1)                             # max sim each client sees
    for i in range(n):
        for j in range(n):
            if i != j and v[j] > eps and v[i] < v[j]:
                cs[i, j] *= v[i] / v[j]
    v = cs.max(axis=1)
    alpha = np.clip(1.0 - v, 0.0, 1.0)
    amax = float(alpha.max())
    if amax > eps:
        alpha = alpha / amax
    a_eps = np.clip(alpha, eps, 1 - eps)
    alpha = kappa * (np.log(a_eps / (1 - a_eps)) + 0.5)
    alpha = np.clip(alpha, 0.0, 1.0)
    s = alpha.sum()
    if s <= eps:
        w = np.ones(n) / n
    else:
        w = alpha / s
    stacked = np.stack([u.astype(np.float64) for u in updates], axis=0)
    return (w.reshape(-1, *([1] * (stacked.ndim - 1))) * stacked).sum(axis=0).astype(np.float32)


_DISPATCH = {
    "fedavg": fedavg,
    "median": fedmedian,
    "fedmedian": fedmedian,
    "trimmed_mean": trimmed_mean,
    "krum": krum,
    "multi_krum": multi_krum,
    "bulyan": bulyan,
    "foolsgold": foolsgold,
}


def aggregate(method: str, updates: Sequence[np.ndarray],
              weights: Sequence[float] = None, **kwargs) -> np.ndarray:
    """Dispatch to the requested aggregation method.

    Unknown methods raise ``ValueError`` so configuration typos surface
    immediately instead of silently degrading to FedAvg.
    """
    key = (method or "fedavg").lower()
    if key not in _DISPATCH:
        raise ValueError(
            f"Unknown aggregation method '{method}'. "
            f"Choose from: {sorted(_DISPATCH)}")
    fn = _DISPATCH[key]
    if weights is None:
        weights = [1.0 / max(len(updates), 1)] * len(updates)
    return fn(updates, weights, **kwargs)

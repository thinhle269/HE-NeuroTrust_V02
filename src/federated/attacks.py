"""State-of-the-art *coordinated* model-poisoning attacks.

The simple per-client attacks in :mod:`src.federated.client`
(``label_flip``, ``sign_flip``, ``gaussian_noise``) are *uncoordinated*:
each malicious client corrupts its own update independently, so an honest
majority absorbs them at low Byzantine fractions.  Real adversaries are
smarter.  This module implements four attacks that are explicitly crafted
to defeat robust aggregators by staying *inside* the distribution of
benign updates - so distance / rank based defences (Krum, Median,
Trimmed-Mean, Bulyan) cannot flag them, and honest-majority averaging no
longer neutralises them even at 20-30% Byzantine.

Threat model.  The adversary controls the ``f`` malicious clients and can
see their honest updates plus an estimate of the benign mean and variance
(standard "full-knowledge" assumption used by all four source papers).
Each attack replaces every malicious client's plaintext update with a
single crafted vector, computed by the coordinating adversary from the
benign set.

Implemented
-----------
* ``alie``      - "A Little Is Enough" (Baruch et al., NeurIPS 2019).
* ``ipm``       - Inner Product Manipulation (Xie et al., UAI 2020).
* ``min_max``   - Min-Max agnostic attack (Shejwalkar & Houmansadr, NDSS 2021).
* ``min_sum``   - Min-Sum agnostic attack (Shejwalkar & Houmansadr, NDSS 2021).

Each function takes the stacked benign updates and returns the single
crafted malicious vector; :func:`apply_coordinated_attack` wires them into
the list of :class:`ClientUpdate` objects.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np
from scipy.stats import norm

COORDINATED = {"alie", "ipm", "min_max", "min_sum"}


def _benign_stats(benign: np.ndarray):
    """Return (mean, std) over the benign update matrix (n_benign, d)."""
    mu = benign.mean(axis=0)
    sigma = benign.std(axis=0)
    return mu, sigma


def alie(benign: np.ndarray, n_total: int, n_malicious: int) -> np.ndarray:
    """A Little Is Enough (Baruch et al. 2019).

    Shifts the benign mean by ``z_max`` standard deviations, where
    ``z_max`` is the largest deviation that still keeps the crafted update
    statistically indistinguishable from the benign population given the
    number of supporters an aggregator would require.
    """
    mu, sigma = _benign_stats(benign)
    n, m = int(n_total), int(n_malicious)
    s = np.floor(n / 2 + 1) - m
    s = max(s, 1)
    denom = max(n - m, 1)
    frac = float(np.clip((n - m - s) / denom, 1e-6, 1 - 1e-6))
    z_max = float(norm.ppf(frac))
    z_max = z_max if np.isfinite(z_max) else 0.0
    return (mu - z_max * sigma).astype(np.float32)


def ipm(benign: np.ndarray, epsilon: float = 0.5) -> np.ndarray:
    """Inner Product Manipulation (Xie et al. 2020).

    Sets the malicious update to a negatively-scaled benign mean so that
    the aggregated update has a negative inner product with the true
    gradient (i.e. it points *away* from the descent direction).
    """
    mu, _ = _benign_stats(benign)
    return (-float(epsilon) * mu).astype(np.float32)


def _pairwise_max_dist(x: np.ndarray) -> float:
    """Max pairwise Euclidean distance among rows of ``x``."""
    sq = np.sum(x * x, axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (x @ x.T)
    d2 = np.maximum(d2, 0.0)
    return float(np.sqrt(d2.max())) if x.shape[0] > 1 else 0.0


def min_max(benign: np.ndarray, perturbation: str = "std") -> np.ndarray:
    """Min-Max agnostic attack (Shejwalkar & Houmansadr 2021).

    Finds the largest scale ``gamma`` such that the crafted update's
    *maximum* distance to any benign update does not exceed the maximum
    benign-to-benign distance, guaranteeing it survives distance filters.
    """
    mu, sigma = _benign_stats(benign)
    p = _perturbation_dir(mu, sigma, perturbation)
    threshold = _pairwise_max_dist(benign)
    gamma = _binary_search_gamma(
        benign, mu, p,
        lambda cand: max(np.linalg.norm(cand - b) for b in benign),
        threshold,
    )
    return (mu + gamma * p).astype(np.float32)


def min_sum(benign: np.ndarray, perturbation: str = "std") -> np.ndarray:
    """Min-Sum agnostic attack (Shejwalkar & Houmansadr 2021).

    Like Min-Max but bounds the *sum* of squared distances to the benign
    updates by the maximum benign sum-of-squared-distances.
    """
    mu, sigma = _benign_stats(benign)
    p = _perturbation_dir(mu, sigma, perturbation)
    sums = []
    for i in range(benign.shape[0]):
        sums.append(float(np.sum(np.sum((benign - benign[i]) ** 2, axis=1))))
    threshold = max(sums) if sums else 0.0
    gamma = _binary_search_gamma(
        benign, mu, p,
        lambda cand: float(np.sum(np.sum((benign - cand) ** 2, axis=1))),
        threshold,
    )
    return (mu + gamma * p).astype(np.float32)


def _perturbation_dir(mu: np.ndarray, sigma: np.ndarray, kind: str) -> np.ndarray:
    kind = (kind or "std").lower()
    if kind == "std":
        return -sigma
    if kind == "sign":
        return -np.sign(mu)
    if kind == "mean":
        return -mu
    raise ValueError(f"Unknown perturbation direction: {kind}")


def _binary_search_gamma(benign, mu, p, objective, threshold,
                         hi: float = 100.0, iters: int = 25) -> float:
    """Largest gamma in [0, hi] with ``objective(mu + gamma*p) <= threshold``."""
    lo, best = 0.0, 0.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        cand = mu + mid * p
        if objective(cand) <= threshold:
            best = mid
            lo = mid
        else:
            hi = mid
    return best


def craft_malicious(method: str, benign: np.ndarray,
                    n_total: int, n_malicious: int,
                    epsilon: float = 0.5,
                    perturbation: str = "std") -> np.ndarray:
    key = (method or "").lower()
    if key == "alie":
        return alie(benign, n_total, n_malicious)
    if key == "ipm":
        return ipm(benign, epsilon=epsilon)
    if key == "min_max":
        return min_max(benign, perturbation=perturbation)
    if key == "min_sum":
        return min_sum(benign, perturbation=perturbation)
    raise ValueError(f"Unknown coordinated attack '{method}'. "
                     f"Choose from {sorted(COORDINATED)}")


def apply_coordinated_attack(client_updates: Sequence, method: str,
                             epsilon: float = 0.5,
                             perturbation: str = "std") -> int:
    """Replace every malicious client's ``flat_update`` with a crafted vector.

    Operates in place on the ``ClientUpdate`` objects.  Returns the number
    of malicious clients whose update was overwritten.  If there are no
    benign updates to craft from (all clients malicious) the updates are
    left unchanged.
    """
    benign = np.stack([u.flat_update for u in client_updates if not u.is_malicious], axis=0) \
        if any(not u.is_malicious for u in client_updates) else None
    malicious = [u for u in client_updates if u.is_malicious]
    if benign is None or benign.shape[0] == 0 or not malicious:
        return 0
    crafted = craft_malicious(method, benign,
                              n_total=len(client_updates),
                              n_malicious=len(malicious),
                              epsilon=epsilon, perturbation=perturbation)
    crafted = np.nan_to_num(crafted, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    for u in malicious:
        u.flat_update = crafted.copy()
    return len(malicious)

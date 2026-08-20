"""Differentiable neuro-fuzzy (ANFIS-style) trust engine.

This module upgrades the hand-designed Mamdani engine in
:mod:`src.fuzzy.trust_engine` with a *learnable* Adaptive Neuro-Fuzzy
Inference System (ANFIS).  The design is ported from the authors' prior
work V-TrustFL (Le et al.), adapted from a 2-input / 25-rule head for
continuous mobile authentication to the 3-input trust signals used by the
IDS pipeline here.

Motivation
----------
A recurring Q1 reviewer objection to fuzzy-logic trust systems is *"the
membership functions are hand-picked"*.  The Mamdani engine answers this
with a three-views derivation of a single threshold, but the *shapes* of
the trapezoidal sets themselves are still fixed a priori.  The neuro-fuzzy
head removes that objection entirely: the membership-function parameters
(Gaussian means and spreads) and the rule consequences are **trainable**
and are calibrated from data, so the partition self-adapts to the
deployment's actual honest/malicious attestation distribution.

Architecture (mirrors V-TrustFL Eqs. 9-12)
------------------------------------------
Inputs: the three trust signals produced by
:meth:`FuzzyTrustEngine.build_features` -
``cosine_sim`` in [-1, 1], ``loss_improvement`` in [-1, 1],
``data_volume`` in [0, 1].

1. Fuzzification - each input x is mapped through ``K`` Gaussian MFs with
   trainable mean ``m`` and spread ``s``:
       mu(x; m, s) = exp( -0.5 * ((x - m) / (|s| + eps))^2 )
2. Rule firing - the product T-norm over the three inputs gives
   ``K^3`` differentiable rules:
       f_{ijk} = mu^cos_i * mu^loss_j * mu^vol_k
3. Defuzzification - trainable consequence weights ``w`` (sigmoid-bounded
   to [0, 1]) combined by a firing-strength-weighted average:
       T_fuzzy = sum_r f_r * sigma(w_r) / (sum_r f_r + eps)
4. Residual connection - a learnable ``alpha`` blends a cheap linear
   "base" trust with the fuzzy-refined score:
       trust = sigma(alpha) * base + (1 - sigma(alpha)) * T_fuzzy

The whole forward pass is differentiable, so the engine is trained by
gradient descent (Adam + binary cross-entropy) on labelled attestation
triples via :meth:`NeuroFuzzyTrustEngine.fit`.

Interface parity
----------------
The public surface mirrors :class:`FuzzyTrustEngine` so the two engines are
drop-in interchangeable in :class:`src.federated.server.FederatedServer`:
``score(features)``, ``score_many(features)`` and the shared static
``build_features`` (re-exported from the Mamdani module).
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from .trust_engine import TrustFeatures, FuzzyTrustEngine

_EPS = 1e-6


class _ANFISHead(nn.Module):
    """The differentiable ANFIS forward pass as a torch Module."""

    def __init__(self, n_inputs: int = 3, n_mf: int = 5,
                 input_lows: Sequence[float] = (-1.0, -1.0, 0.0),
                 input_highs: Sequence[float] = (1.0, 1.0, 1.0)):
        super().__init__()
        self.n_inputs = int(n_inputs)
        self.n_mf = int(n_mf)
        lows = torch.tensor(input_lows, dtype=torch.float32)
        highs = torch.tensor(input_highs, dtype=torch.float32)

        means = torch.stack([
            torch.linspace(float(lows[i]), float(highs[i]), self.n_mf)
            for i in range(self.n_inputs)
        ], dim=0)  # (n_inputs, n_mf)
        span = (highs - lows) / max(self.n_mf - 1, 1)
        spreads = span.unsqueeze(1).repeat(1, self.n_mf)  # (n_inputs, n_mf)

        self.means = nn.Parameter(means)
        self.log_spreads = nn.Parameter(torch.log(spreads + 0.1))

        n_rules = self.n_mf ** self.n_inputs
        self.rule_logits = nn.Parameter(torch.zeros(n_rules))

        self.base = nn.Linear(self.n_inputs, 1)
        with torch.no_grad():
            self.base.weight.copy_(torch.tensor([[1.5, 1.0, 0.3]]))
            self.base.bias.zero_()

        self.alpha = nn.Parameter(torch.tensor(0.0))

    def _memberships(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Return per-input MF activations, each shape (B, n_mf)."""
        spreads = torch.exp(self.log_spreads)  # (n_inputs, n_mf)
        mus = []
        for i in range(self.n_inputs):
            xi = x[:, i:i + 1]                     # (B, 1)
            m = self.means[i].unsqueeze(0)         # (1, n_mf)
            s = spreads[i].unsqueeze(0)            # (1, n_mf)
            mu = torch.exp(-0.5 * ((xi - m) / (s.abs() + _EPS)) ** 2)
            mus.append(mu)
        return mus

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mus = self._memberships(x)                 # list of (B, n_mf)
        f = mus[0].unsqueeze(2).unsqueeze(3) \
            * mus[1].unsqueeze(1).unsqueeze(3) \
            * mus[2].unsqueeze(1).unsqueeze(2)     # (B, K, K, K)
        f = f.reshape(f.shape[0], -1)              # (B, n_rules)
        w = torch.sigmoid(self.rule_logits).unsqueeze(0)  # (1, n_rules)
        t_fuzzy = (f * w).sum(dim=1) / (f.sum(dim=1) + _EPS)   # (B,)

        base = torch.sigmoid(self.base(x)).squeeze(1)          # (B,)
        a = torch.sigmoid(self.alpha)
        trust = a * base + (1.0 - a) * t_fuzzy
        return trust.clamp(0.0, 1.0)


class NeuroFuzzyTrustEngine:
    """Learnable ANFIS trust engine, drop-in for :class:`FuzzyTrustEngine`.

    Before it is fitted the engine already produces sensible scores thanks
    to the informative parameter initialisation (base weights favour high
    cosine similarity and loss improvement).  Calling :meth:`fit` with
    labelled attestation triples calibrates the MFs and rule consequences
    to the deployment's actual attack distribution.
    """

    def __init__(self, cfg_fuzzy=None, n_mf: int = 5, lr: float = 0.01,
                 device: str = "cpu", seed: int = 42):
        self._cfg = cfg_fuzzy
        self.device = torch.device(device)
        self.n_mf = int(n_mf)
        self.lr = float(lr)
        self._seed = int(seed)
        torch.manual_seed(self._seed)
        self.model = _ANFISHead(n_inputs=3, n_mf=self.n_mf).to(self.device)
        self._fitted = False

    @staticmethod
    def _to_matrix(features: Iterable[TrustFeatures]) -> np.ndarray:
        rows = []
        for f in features:
            c = f.cosine_sim if f.cosine_sim == f.cosine_sim else -1.0
            l = f.loss_improvement if f.loss_improvement == f.loss_improvement else 0.0
            v = f.data_volume if f.data_volume == f.data_volume else 0.0
            rows.append([float(np.clip(c, -1, 1)),
                         float(np.clip(l, -1, 1)),
                         float(np.clip(v, 0, 1))])
        return np.asarray(rows, dtype=np.float32).reshape(-1, 3)

    def fit(self, features: Sequence[TrustFeatures], labels: Sequence[float],
            epochs: int = 300, verbose: bool = False) -> "NeuroFuzzyTrustEngine":
        """Calibrate the ANFIS on labelled attestation triples.

        ``labels`` are soft trust targets in [0, 1] (1 = honest, 0 = malicious).
        """
        X = torch.from_numpy(self._to_matrix(features)).to(self.device)
        y = torch.tensor(np.asarray(labels, dtype=np.float32).reshape(-1),
                         device=self.device)
        if X.shape[0] == 0:
            return self
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.BCELoss()
        self.model.train()
        for ep in range(int(epochs)):
            opt.zero_grad(set_to_none=True)
            pred = self.model(X)
            loss = loss_fn(pred.clamp(_EPS, 1 - _EPS), y)
            loss.backward()
            opt.step()
            if verbose and (ep % 50 == 0 or ep == epochs - 1):
                print(f"[neuro-fuzzy] epoch {ep:3d}  bce={float(loss):.4f}")
        self.model.eval()
        self._fitted = True
        return self

    def score(self, features: TrustFeatures) -> float:
        return float(self.score_many([features])[0])

    def score_many(self, features: Iterable[TrustFeatures]) -> np.ndarray:
        feats = list(features)
        if not feats:
            return np.zeros(0, dtype=np.float64)
        X = torch.from_numpy(self._to_matrix(feats)).to(self.device)
        self.model.eval()
        with torch.no_grad():
            out = self.model(X).cpu().numpy().astype(np.float64)
        return out

    build_features = staticmethod(FuzzyTrustEngine.build_features)

    @property
    def is_fitted(self) -> bool:
        return self._fitted

"""Mamdani fuzzy inference for client trust scoring.

Inputs (per client, per round):

* ``cosine_sim``        - cosine similarity between this client's update
                          direction and the *median* update across clients.
                          Honest clients tend to point in similar directions
                          while colluding adversaries diverge.
* ``loss_improvement``  - relative drop in local training loss
                          (``(loss_before - loss_after) / max(|loss_before|, 1e-6)``).
                          A negative or near-zero value is suspicious because
                          it may indicate the client is intentionally producing
                          poor updates.
* ``data_volume``       - share of total samples that this client contributes,
                          normalised to ``[0, 1]``.  Larger data sets should be
                          weighted more, but only when the other two signals
                          are also healthy.

Output:

* ``trust``             - real number in ``[0, 1]`` used (i) to weight the
                          ciphertext during homomorphic aggregation and
                          (ii) as the input to the Zero-Trust policy engine
                          for accept / reject decisions.

The rule base is the classic Mamdani style with ``min`` for AND, ``max`` for
the implication aggregation, and centroid defuzzification - all provided by
``scikit-fuzzy``.  Trapezoidal membership functions are defined in the
configuration file so they can be tuned per dataset without touching code.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


@dataclass
class TrustFeatures:
    cosine_sim: float
    loss_improvement: float
    data_volume: float


def _safe_clip(value: float, lo: float, hi: float) -> float:
    if value != value:  # NaN check
        return lo
    return float(min(hi, max(lo, value)))


class FuzzyTrustEngine:
    """Mamdani FIS that converts raw client-update statistics into a trust score."""

    def __init__(self, cfg_fuzzy):
        self._cfg = cfg_fuzzy
        self._cosine = ctrl.Antecedent(np.linspace(-1.0, 1.0, 401), "cosine_sim")
        self._loss = ctrl.Antecedent(np.linspace(-1.0, 1.0, 401), "loss_improvement")
        self._volume = ctrl.Antecedent(np.linspace(0.0, 1.0, 201), "data_volume")
        self._trust = ctrl.Consequent(np.linspace(0.0, 1.0, 201), "trust")

        self._build_memberships()
        rules = self._build_rules()
        self._system = ctrl.ControlSystem(rules)
        self.last_firing: Dict[str, float] = {}

    def _build_memberships(self) -> None:
        f_inputs = self._cfg.inputs
        c = self._cosine
        c["low"] = fuzz.trapmf(c.universe, f_inputs.cosine_sim.sets.low)
        c["medium"] = fuzz.trapmf(c.universe, f_inputs.cosine_sim.sets.medium)
        c["high"] = fuzz.trapmf(c.universe, f_inputs.cosine_sim.sets.high)
        l = self._loss
        l["worse"] = fuzz.trapmf(l.universe, f_inputs.loss_improvement.sets.worse)
        l["neutral"] = fuzz.trapmf(l.universe, f_inputs.loss_improvement.sets.neutral)
        l["better"] = fuzz.trapmf(l.universe, f_inputs.loss_improvement.sets.better)
        v = self._volume
        v["small"] = fuzz.trapmf(v.universe, f_inputs.data_volume.sets.small)
        v["medium"] = fuzz.trapmf(v.universe, f_inputs.data_volume.sets.medium)
        v["large"] = fuzz.trapmf(v.universe, f_inputs.data_volume.sets.large)
        t = self._trust
        out = self._cfg.output.trust
        t["untrusted"] = fuzz.trapmf(t.universe, out.sets.untrusted)
        t["suspicious"] = fuzz.trapmf(t.universe, out.sets.suspicious)
        t["trusted"] = fuzz.trapmf(t.universe, out.sets.trusted)
        t.defuzzify_method = "centroid"

    def _build_rules(self) -> List[ctrl.Rule]:
        c, l, v, t = self._cosine, self._loss, self._volume, self._trust
        rules: List[ctrl.Rule] = [
            ctrl.Rule(c["low"] & l["worse"] & v["large"], t["untrusted"]),
            ctrl.Rule(c["low"] & l["worse"] & v["medium"], t["untrusted"]),
            ctrl.Rule(c["low"] & l["worse"] & v["small"], t["untrusted"]),
            ctrl.Rule(c["low"] & l["neutral"] & v["large"], t["untrusted"]),
            ctrl.Rule(c["low"] & l["neutral"] & v["medium"], t["untrusted"]),
            ctrl.Rule(c["low"] & l["neutral"] & v["small"], t["untrusted"]),
            ctrl.Rule(c["low"] & l["better"] & v["large"], t["suspicious"]),
            ctrl.Rule(c["low"] & l["better"] & v["medium"], t["suspicious"]),
            ctrl.Rule(c["low"] & l["better"] & v["small"], t["untrusted"]),
            ctrl.Rule(c["medium"] & l["worse"] & v["large"], t["suspicious"]),
            ctrl.Rule(c["medium"] & l["worse"] & v["medium"], t["suspicious"]),
            ctrl.Rule(c["medium"] & l["worse"] & v["small"], t["untrusted"]),
            ctrl.Rule(c["medium"] & l["neutral"] & v["large"], t["suspicious"]),
            ctrl.Rule(c["medium"] & l["neutral"] & v["medium"], t["suspicious"]),
            ctrl.Rule(c["medium"] & l["neutral"] & v["small"], t["suspicious"]),
            ctrl.Rule(c["medium"] & l["better"] & v["large"], t["trusted"]),
            ctrl.Rule(c["medium"] & l["better"] & v["medium"], t["trusted"]),
            ctrl.Rule(c["medium"] & l["better"] & v["small"], t["suspicious"]),
            ctrl.Rule(c["high"] & l["worse"] & v["large"], t["suspicious"]),
            ctrl.Rule(c["high"] & l["worse"] & v["medium"], t["suspicious"]),
            ctrl.Rule(c["high"] & l["worse"] & v["small"], t["untrusted"]),
            ctrl.Rule(c["high"] & l["neutral"] & v["large"], t["trusted"]),
            ctrl.Rule(c["high"] & l["neutral"] & v["medium"], t["trusted"]),
            ctrl.Rule(c["high"] & l["neutral"] & v["small"], t["suspicious"]),
            ctrl.Rule(c["high"] & l["better"] & v["large"], t["trusted"]),
            ctrl.Rule(c["high"] & l["better"] & v["medium"], t["trusted"]),
            ctrl.Rule(c["high"] & l["better"] & v["small"], t["trusted"]),
        ]
        return rules

    def score(self, features: TrustFeatures) -> float:
        """Return defuzzified trust score in ``[0, 1]`` for a single client."""
        sim = ctrl.ControlSystemSimulation(self._system)
        sim.input["cosine_sim"] = _safe_clip(features.cosine_sim, -1.0, 1.0)
        sim.input["loss_improvement"] = _safe_clip(features.loss_improvement, -1.0, 1.0)
        sim.input["data_volume"] = _safe_clip(features.data_volume, 0.0, 1.0)
        sim.compute()
        score = float(sim.output.get("trust", 0.5))
        try:
            self.last_firing = {
                name: float(self._trust.terms[name].membership_value[sim])
                for name in ("untrusted", "suspicious", "trusted")
            }
        except Exception:
            self.last_firing = {}
        return _safe_clip(score, 0.0, 1.0)

    def score_many(self, features: Iterable[TrustFeatures]) -> np.ndarray:
        return np.array([self.score(f) for f in features], dtype=np.float64)

    @staticmethod
    def build_features(updates: Sequence[np.ndarray],
                       local_losses_before: Sequence[float],
                       local_losses_after: Sequence[float],
                       data_sizes: Sequence[int],
                       reference=None) -> List[TrustFeatures]:
        """Compute the three fuzzy features for each client in one pass.

        The directional feature ``cos_i`` is the cosine similarity between a
        client's update and a *reference direction*.  When ``reference`` is
        supplied (the previous accepted global aggregate - a public vector that
        every client already holds from the broadcast model history), cos_i is
        computed against it, so the trust signal needs no plaintext view of the
        other clients' updates and can be produced client-side.  When
        ``reference`` is ``None`` (e.g. the first round, before any aggregate
        exists) it falls back to the element-wise median of the round's updates
        as a one-round bootstrap.
        """
        if not updates:
            return []
        flat = [u.reshape(-1).astype(np.float64) for u in updates]
        if reference is not None:
            ref = np.asarray(reference, dtype=np.float64).reshape(-1)
            if np.linalg.norm(ref) < 1e-9:
                ref = np.median(np.stack(flat, axis=0), axis=0)
        else:
            ref = np.median(np.stack(flat, axis=0), axis=0)
        ref_norm = np.linalg.norm(ref) + 1e-12
        cos_sims = []
        for f in flat:
            n = np.linalg.norm(f) + 1e-12
            cos_sims.append(float(np.dot(f, ref) / (n * ref_norm)))

        improvements = []
        for before, after in zip(local_losses_before, local_losses_after):
            denom = max(abs(before), 1e-6)
            improvements.append(_safe_clip((before - after) / denom, -1.0, 1.0))

        total = float(sum(data_sizes)) or 1.0
        volumes = [d / total for d in data_sizes]
        return [TrustFeatures(c, l, v) for c, l, v in zip(cos_sims, improvements, volumes)]

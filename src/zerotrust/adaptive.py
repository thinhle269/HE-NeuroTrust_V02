"""Adaptive controllers for the Zero-Trust hyperparameters.

In a real deployment the threat profile does not stay fixed: an attacker
may stay quiet during the warm-up phase, then turn on a sign-flip attack,
then switch to a label-flip attack to evade a defence calibrated for sign-
flip.  A static threshold tau and EMA weight alpha cannot follow these
regime changes - the operator either tunes for the worst case (sacrificing
accuracy in benign periods) or the average case (vulnerable to peaks).

We therefore implement two principled, *observable-signal-only* controllers
that update tau and alpha at the end of every round.  Both are exposed
behind a config flag so the manuscript can compare static and adaptive
variants on identical data.

Design principles
-----------------
* **Read-only signals.**  The controllers consume quantities the server
  already has (the round report - val_macro_f1, trust scores, rejection
  rate, n_malicious_rejected).  No new observability requirement.
* **Bounded action.**  tau is clipped to ``[tau_min, tau_max]``; alpha
  to ``[alpha_min, alpha_max]``.  Default bounds [0.10, 0.50] cover the
  empirically-safe region from the ablation - the controller cannot
  drive the policy into either deadlock (tau >= 0.50) or no-defence
  (tau < 0.10).
* **Damped response.**  We use multiplicative damping (k_up / k_down)
  rather than full PID so the controllers are *interpretable* in the
  manuscript without a tuning loop of their own.

Algorithmic summary
-------------------
* :class:`AdaptiveTauController` raises tau when validation macro-F1 is
  dropping (proxy for "under active attack") and lowers it when macro-F1
  is recovering (proxy for "attack subsided / over-rejection").
* :class:`AdaptiveAlphaController` raises alpha when the per-client
  trust scores change abruptly between rounds (proxy for "client just
  switched strategy - forget history") and lowers it when scores are
  stable (proxy for "smooth signal - history is useful").
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional, Sequence


@dataclass
class AdaptiveTauController:
    """PI-style controller that adapts tau to the validation trajectory.

    State carried across rounds:
        ``_f1_history`` - sliding window of recent val_macro_f1 values.
        ``_tau`` - current threshold; updated in place.

    Parameters (all in config):
        tau_init / tau_min / tau_max - initial value and clipping bounds.
        f1_window - number of rounds in the smoothing window (default 3).
        k_up - tau increment when an attack signal is detected.
        k_down - tau decrement when the signal fades.
        drop_trigger - relative F1 drop (current vs older window mean)
                       that triggers ``k_up``.
        recover_trigger - relative F1 rise that triggers ``k_down``.
    """
    tau_init: float = 0.20
    tau_min: float = 0.10
    tau_max: float = 0.45
    f1_window: int = 3
    k_up: float = 0.05
    k_down: float = 0.02
    drop_trigger: float = 0.02
    recover_trigger: float = 0.02

    _tau: float = field(init=False, default=0.0)
    _f1_history: Deque[float] = field(init=False,
                                      default_factory=lambda: deque(maxlen=64))
    _log: list = field(init=False, default_factory=list)

    def __post_init__(self):
        self._tau = float(self.tau_init)
        if not (0.0 <= self.tau_min <= self.tau_max <= 1.0):
            raise ValueError("tau_min/tau_max must satisfy 0 <= min <= max <= 1")

    @property
    def tau(self) -> float:
        return float(self._tau)

    @property
    def history(self) -> list:
        return list(self._log)

    def update(self, val_macro_f1: float, round_idx: int) -> float:
        """Return the tau to use *for the next round*.

        Call this after the round's evaluation.  The new tau takes effect
        when ``ZeroTrustPolicy`` re-reads the controller's value at the
        start of the following round.
        """
        f1 = float(val_macro_f1) if val_macro_f1 == val_macro_f1 else 0.0
        self._f1_history.append(f1)
        signal = "warmup"
        if len(self._f1_history) >= 2 * self.f1_window:
            recent = sum(list(self._f1_history)[-self.f1_window:]) / self.f1_window
            older = sum(list(self._f1_history)[-2 * self.f1_window:-self.f1_window]) / self.f1_window
            if recent < older - self.drop_trigger:
                self._tau = min(self._tau + self.k_up, self.tau_max)
                signal = "tighten"
            elif recent > older + self.recover_trigger:
                self._tau = max(self._tau - self.k_down, self.tau_min)
                signal = "relax"
            else:
                signal = "hold"
        self._log.append({
            "round_idx": int(round_idx),
            "val_macro_f1": float(f1),
            "tau_next": float(self._tau),
            "signal": signal,
        })
        return self._tau


@dataclass
class AdaptiveAlphaController:
    """Adapts the EMA weight alpha based on per-client trust stability.

    The intuition: if the per-client trust scores barely change from one
    round to the next, the signal is stationary and we should *trust the
    history* - a low alpha (heavy smoothing) is appropriate.  If the
    scores jump significantly between rounds, the client population is
    behaviour-shifting and we need a *higher* alpha so the EMA can
    catch up quickly.

    The "change" signal is the mean absolute change in raw_trust per
    client over the last two rounds:
        delta = mean_i |raw_trust_i^(r) - raw_trust_i^(r-1)|

    delta near 0 -> stationary -> alpha = alpha_min
    delta large  -> shifting   -> alpha = alpha_max
    """
    alpha_init: float = 0.20
    alpha_min: float = 0.20
    alpha_max: float = 0.70
    change_low: float = 0.05    # delta below this -> consider stationary
    change_high: float = 0.20   # delta above this -> consider shifting
    _alpha: float = field(init=False, default=0.0)
    _prev_trust: Optional[Dict[int, float]] = field(init=False, default=None)
    _log: list = field(init=False, default_factory=list)

    def __post_init__(self):
        self._alpha = float(self.alpha_init)
        if not (0.0 <= self.alpha_min <= self.alpha_max <= 1.0):
            raise ValueError("alpha bounds must satisfy 0 <= min <= max <= 1")
        if self.change_low > self.change_high:
            raise ValueError("change_low must be <= change_high")

    @property
    def alpha(self) -> float:
        return float(self._alpha)

    @property
    def history(self) -> list:
        return list(self._log)

    def update(self, raw_trust: Dict[int, float], round_idx: int) -> float:
        cur = {int(c): float(v) for c, v in raw_trust.items()}
        signal = "warmup"
        delta = float("nan")
        if self._prev_trust is not None and len(self._prev_trust) > 0:
            shared = set(cur.keys()) & set(self._prev_trust.keys())
            if shared:
                diffs = [abs(cur[c] - self._prev_trust[c]) for c in shared]
                delta = sum(diffs) / len(diffs)
                if delta < self.change_low:
                    self._alpha = self.alpha_min
                    signal = "smooth"
                elif delta > self.change_high:
                    self._alpha = self.alpha_max
                    signal = "react"
                else:
                    t = (delta - self.change_low) / (self.change_high - self.change_low)
                    self._alpha = self.alpha_min + t * (self.alpha_max - self.alpha_min)
                    signal = "mixed"
        self._prev_trust = cur
        self._log.append({
            "round_idx": int(round_idx),
            "trust_delta": float(delta),
            "alpha_next": float(self._alpha),
            "signal": signal,
        })
        return self._alpha


def build_adaptive_controllers(cfg) -> Dict[str, object]:
    """Factory: read the adaptive subsection of ``cfg.zero_trust`` and
    instantiate whichever controllers are requested.  Returns an empty
    dict if adaptive mode is disabled.
    """
    if not hasattr(cfg, "zero_trust"):
        return {}
    zt = cfg.zero_trust
    adapt = zt.get("adaptive") if hasattr(zt, "get") else None
    if not adapt:
        return {}
    out: Dict[str, object] = {}
    tau_cfg = adapt.get("tau") if hasattr(adapt, "get") else None
    if tau_cfg and bool(tau_cfg.get("enabled", False)):
        out["tau"] = AdaptiveTauController(
            tau_init=float(tau_cfg.get("init", zt.trust_threshold)),
            tau_min=float(tau_cfg.get("min", 0.10)),
            tau_max=float(tau_cfg.get("max", 0.45)),
            f1_window=int(tau_cfg.get("f1_window", 3)),
            k_up=float(tau_cfg.get("k_up", 0.05)),
            k_down=float(tau_cfg.get("k_down", 0.02)),
            drop_trigger=float(tau_cfg.get("drop_trigger", 0.02)),
            recover_trigger=float(tau_cfg.get("recover_trigger", 0.02)),
        )
    alpha_cfg = adapt.get("alpha") if hasattr(adapt, "get") else None
    if alpha_cfg and bool(alpha_cfg.get("enabled", False)):
        out["alpha"] = AdaptiveAlphaController(
            alpha_init=float(alpha_cfg.get("init", zt.get("ema_alpha", 0.2))),
            alpha_min=float(alpha_cfg.get("min", 0.20)),
            alpha_max=float(alpha_cfg.get("max", 0.70)),
            change_low=float(alpha_cfg.get("change_low", 0.05)),
            change_high=float(alpha_cfg.get("change_high", 0.20)),
        )
    return out

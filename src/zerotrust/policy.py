"""Zero-Trust policy engine.

Embodies the NIST SP 800-207 *never-trust-always-verify* principle on top of
the fuzzy trust score:

* Every client is re-evaluated **every round** - past acceptance does not
  imply future acceptance.
* A short trust history (sliding window) damps single-round noise but does
  *not* override a sudden trust collapse - we use EMA with explicit decay
  for previously-rejected clients so attackers cannot trivially earn back
  weight by behaving for one round.
* Clients whose smoothed trust falls below :attr:`threshold` are
  **rejected**, i.e. excluded from the homomorphic aggregation in that round
  (their weight becomes 0).
* Surviving clients have their normalised trust used as the aggregation
  weight.

Graceful-degradation safeguards
-------------------------------
A purely-multiplicative ``reject_decay`` cascade can drive a once-
mis-rejected honest client into a *permanent ban* (decay^streak → 0
exponentially).  Once enough honest clients are stuck in that state the
server has nothing to aggregate and training freezes.  We saw this in
empirical runs (full_system, seed 42: rejection cascade after round 8 ->
all clients rejected from round 16 onwards -> loss stuck).

We address this with two complementary safety knobs, both configurable:

* ``max_reject_streak`` caps the multiplicative penalty - after the cap is
  reached the penalty *plateaus* instead of decaying to zero, so a client
  that genuinely rehabilitates can still rejoin.
* ``min_accept_fraction`` enforces a participation floor.  After the
  threshold-based rejection step, if fewer than ``ceil(min_accept_fraction *
  num_clients)`` are accepted we promote the *top-k* by smoothed trust
  regardless of the threshold.  This guarantees that the aggregation step
  is *always* fed at least one update, preventing the system from getting
  stuck on a stale global model.

Both of these knobs can be set to 0 / 1 to recover the original strict
behaviour for ablation purposes.

The class is deliberately stateful: it keeps a per-client history that the
server feeds back in each round.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Sequence


@dataclass
class PolicyDecision:
    """Output of a single-round policy evaluation."""
    accepted: List[int]
    rejected: List[int]
    weights: Dict[int, float]            # only contains accepted clients
    smoothed_trust: Dict[int, float]     # contains *all* clients
    raw_trust: Dict[int, float]          # contains *all* clients
    forced_inclusions: List[int] = None  # accepted *only* because of the floor


class ZeroTrustPolicy:
    def __init__(self, threshold: float = 0.4, history_window: int = 3,
                 reject_decay: float = 0.7, ema_alpha: float = 0.6,
                 max_reject_streak: int = 3,
                 min_accept_fraction: float = 0.5,
                 continuous_verification: bool = True):
        self.threshold = float(threshold)
        self.history_window = int(history_window)
        self.reject_decay = float(reject_decay)
        self.ema_alpha = float(ema_alpha)
        if not (0.0 <= self.ema_alpha <= 1.0):
            raise ValueError("ema_alpha must lie in [0, 1]")
        self.max_reject_streak = max(0, int(max_reject_streak))
        if not (0.0 <= float(min_accept_fraction) <= 1.0):
            raise ValueError("min_accept_fraction must lie in [0, 1]")
        self.min_accept_fraction = float(min_accept_fraction)
        self.continuous_verification = bool(continuous_verification)
        self._history: Dict[int, Deque[float]] = {}
        self._reject_streak: Dict[int, int] = {}

    def evaluate(self, client_ids: Sequence[int],
                 raw_trust: Sequence[float]) -> PolicyDecision:
        if len(client_ids) != len(raw_trust):
            raise ValueError("client_ids/raw_trust size mismatch")

        smoothed: Dict[int, float] = {}
        accepted: List[int] = []
        rejected: List[int] = []

        for cid, score in zip(client_ids, raw_trust):
            hist = self._history.setdefault(cid, deque(maxlen=self.history_window))
            hist.append(float(score))
            smoothed_score = self._ema(hist, alpha=self.ema_alpha)
            streak = self._reject_streak.get(cid, 0)
            effective_streak = min(streak, self.max_reject_streak)
            if effective_streak > 0:
                smoothed_score *= (self.reject_decay ** effective_streak)
            smoothed[cid] = smoothed_score

            if smoothed_score < self.threshold:
                rejected.append(cid)
                self._reject_streak[cid] = streak + 1
            else:
                accepted.append(cid)
                self._reject_streak[cid] = 0

        forced_inclusions: List[int] = []
        n_total = len(client_ids)
        min_k = math.ceil(self.min_accept_fraction * n_total) if self.min_accept_fraction > 0 else 0
        if len(accepted) < min_k:
            ranked = sorted(rejected, key=lambda c: smoothed[c], reverse=True)
            need = min_k - len(accepted)
            for c in ranked[:need]:
                accepted.append(c)
                rejected.remove(c)
                forced_inclusions.append(c)

        weights: Dict[int, float] = {}
        if accepted:
            eps = 1e-3
            adj = {c: max(smoothed[c], eps) for c in accepted}
            total = sum(adj.values())
            weights = {c: w / total for c, w in adj.items()}

        return PolicyDecision(
            accepted=accepted,
            rejected=rejected,
            weights=weights,
            smoothed_trust=smoothed,
            raw_trust={cid: float(s) for cid, s in zip(client_ids, raw_trust)},
            forced_inclusions=forced_inclusions,
        )

    @staticmethod
    def _ema(values: Sequence[float], alpha: float) -> float:
        """Exponential moving average over the (short) trust window.

        ``alpha`` weighs the newest sample more strongly while still letting
        history dampen single-round noise.  Provided by the caller (no
        magic default) so the value is always traceable in the config.
        """
        if not values:
            return 0.0
        ema = float(values[0])
        for v in list(values)[1:]:
            ema = alpha * float(v) + (1.0 - alpha) * ema
        return ema

    def state_dict(self) -> Dict[str, object]:
        return {
            "threshold": self.threshold,
            "history": {c: list(h) for c, h in self._history.items()},
            "reject_streak": dict(self._reject_streak),
        }

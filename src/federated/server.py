"""Federated server orchestrating one experiment scenario.

A single :class:`FederatedServer` instance represents one scenario from
``configs/default.yaml::experiments.scenarios``.  Each scenario specifies:

* ``aggregation``  - one of {fedavg, fedmedian, trimmed_mean, krum, multi_krum}
* ``he`` / ``fuzzy`` / ``zero_trust``  - booleans toggling each defence
* ``malicious``    - whether malicious clients are injected this scenario

This single class therefore supports both literature baselines (FedMedian,
Krum, TrimmedMean, FedProx, plain FedAvg) and our proposed
HE-Fuzzy-ZeroTrust pipeline.  Backward-compatible aliases (``fedavg_he``,
``full_system`` ...) still resolve to the right configuration.

We measure (and persist) the per-round wall-clock cost of each module so
that the paper can report a meaningful overhead breakdown.

Important constraint: Byzantine-robust aggregators (Krum, Median,
TrimmedMean) need plaintext access to client updates and are therefore
*incompatible* with Paillier ciphertexts.  When ``he`` and a non-FedAvg
aggregator are both requested, HE is disabled with a warning - this matches
the paper's discussion of the privacy-vs-robustness trade-off that our
proposed system targets.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..crypto import (EncryptedVector, PaillierContext, DecryptionAuthority,
                      generate_he_parties, secure_aggregate)
from ..crypto.he_paillier import encrypt_vector, decrypt_vector
from ..fuzzy import FuzzyTrustEngine
from ..zerotrust import ZeroTrustPolicy, build_adaptive_controllers
from ..utils.logger import get_logger
from .aggregation import flatten_state, plaintext_aggregate, unflatten_state
from .client import ClientUpdate, FederatedClient
from .robust_aggregation import aggregate as robust_aggregate


_HE_COMPATIBLE_AGGREGATORS = {"fedavg"}

_LEGACY_FLAGS = {
    "fedavg":           ("fedavg",     False, False, False, 0.0),
    "fedavg_he":        ("fedavg",     True,  False, False, 0.0),
    "fedavg_he_fuzzy":  ("fedavg",     True,  True,  False, 0.0),
    "full_system":      ("fedavg",     True,  True,  True,  0.0),
    "full_system_neuro":("fedavg",     True,  True,  True,  0.0),
    "fedmedian":        ("fedmedian",  False, False, False, 0.0),
    "trimmed_mean":     ("trimmed_mean", False, False, False, 0.0),
    "krum":             ("krum",       False, False, False, 0.0),
    "multi_krum":       ("multi_krum", False, False, False, 0.0),
    "bulyan":           ("bulyan",     False, False, False, 0.0),
    "foolsgold":        ("foolsgold",  False, False, False, 0.0),
    "fedprox":          ("fedavg",     False, False, False, 0.01),
}


@dataclass
class RoundReport:
    round_idx: int
    scenario: str
    aggregation_method: str
    use_he: bool
    use_fuzzy: bool
    use_zt: bool
    train_loss_avg: float
    val_loss: float
    val_accuracy: float
    val_macro_f1: float
    val_weighted_f1: float
    accepted_clients: List[int]
    rejected_clients: List[int]
    raw_trust: Dict[int, float]
    smoothed_trust: Dict[int, float]
    aggregation_weights: Dict[int, float]
    n_malicious_total: int
    n_malicious_rejected: int
    time_local_train_sec: float
    time_encrypt_sec: float
    time_aggregate_sec: float
    time_decrypt_sec: float
    time_fuzzy_sec: float
    time_total_sec: float
    he_ciphertext_bytes: int = 0
    per_client: List[dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        d = self.__dict__.copy()
        d["raw_trust"] = {int(k): float(v) for k, v in self.raw_trust.items()}
        d["smoothed_trust"] = {int(k): float(v) for k, v in self.smoothed_trust.items()}
        d["aggregation_weights"] = {int(k): float(v) for k, v in self.aggregation_weights.items()}
        return d


class FederatedServer:
    def __init__(self, cfg, scenario,
                 model: nn.Module,
                 clients: Sequence[FederatedClient],
                 val_loader: DataLoader, test_loader: DataLoader,
                 num_classes: int,
                 device: torch.device,
                 logger=None, project_root: Optional[Path] = None,
                 scenario_cfg: Optional[object] = None):
        self.cfg = cfg
        if hasattr(scenario, "name"):
            scenario_cfg = scenario
            self.scenario = str(scenario.name)
        else:
            self.scenario = str(scenario)
        self.scenario_cfg = scenario_cfg
        self.model = model.to(device)
        self.clients = list(clients)
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.num_classes = int(num_classes)
        self.device = device
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.logger = logger or get_logger(f"fl.server[{self.scenario}]",
                                           self.project_root / cfg.paths.logs_dir,
                                           level=cfg.logging.level)

        agg, he, fz, zt, prox_mu, trim_ratio, byz = self._resolve_flags(scenario_cfg)
        self.aggregation_method = agg
        self.use_fuzzy = bool(fz)
        self.use_zt = bool(zt)
        self.trim_ratio = float(trim_ratio)
        self.num_byzantine = int(byz)
        self.proximal_mu = float(prox_mu)

        if he and self.aggregation_method not in _HE_COMPATIBLE_AGGREGATORS:
            self.logger.warning("[%s] HE requested with aggregation=%s; "
                                "disabling HE (incompatible).",
                                self.scenario, self.aggregation_method)
            he = False
        self.use_he = bool(he)

        self.fuzzy_engine_kind = "mamdani"
        if scenario_cfg is not None and hasattr(scenario_cfg, "get"):
            self.fuzzy_engine_kind = str(scenario_cfg.get("fuzzy_engine", "mamdani"))
        elif hasattr(cfg.fuzzy_logic, "get"):
            self.fuzzy_engine_kind = str(cfg.fuzzy_logic.get("engine", "mamdani"))
        if self.use_fuzzy:
            if self.fuzzy_engine_kind == "neuro":
                from ..fuzzy.neuro_fuzzy import NeuroFuzzyTrustEngine
                self.fuzzy = NeuroFuzzyTrustEngine(cfg.fuzzy_logic, seed=cfg.seed)
                self._neuro_calibrated = False
            else:
                self.fuzzy = FuzzyTrustEngine(cfg.fuzzy_logic)
        else:
            self.fuzzy = None
        _mc = cfg.experiments.malicious_clients
        self.attack_type = str(_mc.attack_type)
        self.attack_epsilon = float(_mc.get("epsilon", 0.5)) if hasattr(_mc, "get") else 0.5
        self.attack_perturbation = str(_mc.get("perturbation", "std")) if hasattr(_mc, "get") else "std"
        _zt_cfg = cfg.zero_trust
        self.zt = (ZeroTrustPolicy(
            threshold=_zt_cfg.trust_threshold,
            history_window=_zt_cfg.history_window,
            reject_decay=_zt_cfg.reject_decay,
            ema_alpha=float(_zt_cfg.get("ema_alpha", 0.6))
                if hasattr(_zt_cfg, "get") else 0.6,
            max_reject_streak=int(_zt_cfg.get("max_reject_streak", 3))
                if hasattr(_zt_cfg, "get") else 3,
            min_accept_fraction=float(_zt_cfg.get("min_accept_fraction", 0.5))
                if hasattr(_zt_cfg, "get") else 0.5,
            continuous_verification=_zt_cfg.continuous_verification,
        ) if self.use_zt else None)
        self.adaptive = build_adaptive_controllers(cfg) if self.use_zt else {}
        self.he_ctx: Optional[PaillierContext] = None
        self.decryptor: Optional[DecryptionAuthority] = None
        if self.use_he:
            self.logger.info("Generating Paillier keypair (key_size=%d) ...",
                             cfg.homomorphic_encryption.key_size)
            self.he_ctx, self.decryptor = generate_he_parties(
                key_size=int(cfg.homomorphic_encryption.key_size),
                quantisation_scale=float(cfg.homomorphic_encryption.quantization_scale),
            )
        self.logger.info(
            "[%s] aggregation=%s he=%s fuzzy=%s zt=%s prox_mu=%g trim_ratio=%g f=%d",
            self.scenario, self.aggregation_method, self.use_he, self.use_fuzzy,
            self.use_zt, self.proximal_mu, self.trim_ratio, self.num_byzantine,
        )

    def _resolve_flags(self, scenario_cfg):
        """Combine per-scenario overrides with legacy name-based defaults."""
        defaults_by_name = _LEGACY_FLAGS.get(self.scenario,
                                             ("fedavg", False, False, False, 0.0))
        agg, he, fz, zt, mu = defaults_by_name
        trim_ratio = 0.2
        byz_default = max(1, int(round(
            float(self.cfg.experiments.malicious_clients.fraction)
            * int(self.cfg.federated.num_clients))))
        byz = byz_default
        if scenario_cfg is not None:
            agg = str(scenario_cfg.get("aggregation", agg))
            he = bool(scenario_cfg.get("he", he))
            fz = bool(scenario_cfg.get("fuzzy", fz))
            zt = bool(scenario_cfg.get("zero_trust", zt))
            mu = float(scenario_cfg.get("proximal_mu", mu))
            trim_ratio = float(scenario_cfg.get("trim_ratio", trim_ratio))
            byz = int(scenario_cfg.get("num_byzantine", byz))
        return agg, he, fz, zt, mu, trim_ratio, byz

    def run(self) -> List[RoundReport]:
        ms_cfg = self.cfg.get("model_selection") if hasattr(self.cfg, "get") else None
        ms_enabled = bool(ms_cfg.get("enabled", True)) if ms_cfg is not None else True
        ms_metric = (str(ms_cfg.get("metric", "macro_f1"))
                     if ms_cfg is not None else "macro_f1")
        ms_higher_is_better = (bool(ms_cfg.get("higher_is_better", True))
                               if ms_cfg is not None else True)
        _attr_map = {
            "accuracy": "val_accuracy", "val_accuracy": "val_accuracy",
            "macro_f1": "val_macro_f1", "val_macro_f1": "val_macro_f1",
            "weighted_f1": "val_weighted_f1", "val_weighted_f1": "val_weighted_f1",
            "loss": "val_loss", "val_loss": "val_loss",
        }
        ms_attr = _attr_map.get(ms_metric, "val_macro_f1")
        if ms_attr == "val_loss" and ms_higher_is_better:
            ms_higher_is_better = False

        best_value = -float("inf") if ms_higher_is_better else float("inf")
        best_state = None
        best_round = -1
        reports: List[RoundReport] = []
        for r in range(int(self.cfg.federated.rounds)):
            report = self._one_round(r)
            reports.append(report)
            self.logger.info(
                "[%s] round %02d/%02d val_loss=%.4f acc=%.4f f1m=%.4f "
                "accept=%d reject=%d mal_rej=%d/%d t_total=%.2fs",
                self.scenario, r + 1, self.cfg.federated.rounds,
                report.val_loss, report.val_accuracy, report.val_macro_f1,
                len(report.accepted_clients), len(report.rejected_clients),
                report.n_malicious_rejected, report.n_malicious_total,
                report.time_total_sec,
            )
            if ms_enabled:
                cur_value = float(getattr(report, ms_attr))
                improved = (cur_value > best_value) if ms_higher_is_better else (cur_value < best_value)
                if improved and np.isfinite(cur_value):
                    best_value = cur_value
                    best_state = {k: v.detach().cpu().clone()
                                  for k, v in self.model.state_dict().items()}
                    best_round = r

            if self.zt is not None and self.adaptive:
                if "tau" in self.adaptive:
                    new_tau = self.adaptive["tau"].update(
                        val_macro_f1=report.val_macro_f1, round_idx=r)
                    self.zt.threshold = float(new_tau)
                if "alpha" in self.adaptive:
                    new_alpha = self.adaptive["alpha"].update(
                        raw_trust=report.raw_trust, round_idx=r)
                    self.zt.ema_alpha = float(new_alpha)
        if ms_enabled and best_state is not None:
            target_device = next(self.model.parameters()).device
            self.model.load_state_dict({k: v.to(target_device) for k, v in best_state.items()})
            self.logger.info(
                "[%s] restored best-validation checkpoint from round %02d "
                "(%s=%.4f); test metrics will be computed on this state.",
                self.scenario, best_round + 1, ms_attr, best_value,
            )
        self.best_round = best_round
        self.best_metric_value = best_value if best_state is not None else float("nan")
        self.best_metric_name = ms_attr
        return reports

    def _one_round(self, round_idx: int) -> RoundReport:
        t_round0 = time.time()
        global_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

        t0 = time.time()
        client_updates: List[ClientUpdate] = []
        sampled = self._sample_clients()
        for client in sampled:
            update = client.train_round(global_state, round_idx=round_idx)
            if not np.isfinite(update.flat_update).all():
                n_bad = int(np.sum(~np.isfinite(update.flat_update)))
                self.logger.warning("[%s] round %d client %d: %d non-finite "
                                    "entries in update; zeroed out",
                                    self.scenario, round_idx,
                                    update.client_id, n_bad)
                update.flat_update = np.nan_to_num(
                    update.flat_update, nan=0.0, posinf=0.0, neginf=0.0
                )
            client_updates.append(update)
        t_local = time.time() - t0

        if not client_updates:
            raise RuntimeError("No clients sampled for round %d" % round_idx)

        from .attacks import COORDINATED, apply_coordinated_attack
        if self.attack_type in COORDINATED:
            n_poisoned = apply_coordinated_attack(
                client_updates, self.attack_type,
                epsilon=self.attack_epsilon,
                perturbation=self.attack_perturbation,
            )
            if n_poisoned and round_idx == 0:
                self.logger.info("[%s] coordinated attack '%s' crafted for %d "
                                 "malicious clients", self.scenario,
                                 self.attack_type, n_poisoned)

        t_enc = 0.0
        he_bytes = 0
        if self.use_he:
            t0 = time.time()
            for u in client_updates:
                u.encrypted_update = encrypt_vector(
                    u.flat_update, self.he_ctx.public_key,
                    scale=self.he_ctx.quantisation_scale, n_jobs=self.he_ctx.n_jobs,
                )
            t_enc = time.time() - t0
            n_bits = self.he_ctx.public_key.n.bit_length() * 2
            ct_count = sum(len(u.encrypted_update) for u in client_updates)
            he_bytes = ct_count * (n_bits // 8)

        t_fuzzy = 0.0
        raw_trust: Dict[int, float] = {}
        if self.use_fuzzy:
            t0 = time.time()
            feats = FuzzyTrustEngine.build_features(
                updates=[u.flat_update for u in client_updates],
                local_losses_before=[u.loss_before for u in client_updates],
                local_losses_after=[u.loss_after for u in client_updates],
                data_sizes=[u.n_samples for u in client_updates],
                reference=getattr(self, "_prev_aggregate", None),
            )
            if self.fuzzy_engine_kind == "neuro" and not getattr(self, "_neuro_calibrated", True):
                self._neuro_calib_X = getattr(self, "_neuro_calib_X", [])
                self._neuro_calib_y = getattr(self, "_neuro_calib_y", [])
                for f, u in zip(feats, client_updates):
                    self._neuro_calib_X.append(f)
                    self._neuro_calib_y.append(0.0 if u.is_malicious else 1.0)
                calib_rounds = int(self.cfg.fuzzy_logic.get("neuro_calib_rounds", 5)) \
                    if hasattr(self.cfg.fuzzy_logic, "get") else 5
                if round_idx + 1 >= calib_rounds and len(set(self._neuro_calib_y)) > 1:
                    self.fuzzy.fit(self._neuro_calib_X, self._neuro_calib_y, epochs=300)
                    self._neuro_calibrated = True
                    self.logger.info("[%s] neuro-fuzzy calibrated on %d labelled "
                                     "attestations after %d rounds", self.scenario,
                                     len(self._neuro_calib_y), round_idx + 1)
            scores = self.fuzzy.score_many(feats)
            raw_trust = {u.client_id: float(s) for u, s in zip(client_updates, scores)}
            t_fuzzy = time.time() - t0
        else:
            raw_trust = {u.client_id: 1.0 for u in client_updates}

        if self.use_zt:
            decision = self.zt.evaluate(
                client_ids=[u.client_id for u in client_updates],
                raw_trust=[raw_trust[u.client_id] for u in client_updates],
            )
            accepted_ids = decision.accepted
            rejected_ids = decision.rejected
            agg_weights = decision.weights
            smoothed = decision.smoothed_trust
        else:
            accepted_ids = [u.client_id for u in client_updates]
            rejected_ids = []
            smoothed = dict(raw_trust)
            mode = self.cfg.federated.weight_aggregation
            if mode == "uniform":
                base = {u.client_id: 1.0 for u in client_updates}
            elif mode == "data_size":
                total_n = sum(u.n_samples for u in client_updates) or 1
                base = {u.client_id: u.n_samples / total_n for u in client_updates}
            else:  # 'trust'
                base = {u.client_id: raw_trust[u.client_id] for u in client_updates}
            total = sum(base.values()) or 1.0
            agg_weights = {cid: w / total for cid, w in base.items()}

        t0 = time.time()
        if accepted_ids:
            accepted_updates = [u for u in client_updates if u.client_id in accepted_ids]
            weights_list = [agg_weights[u.client_id] for u in accepted_updates]
            if self.use_he:
                enc_agg = secure_aggregate(
                    [u.encrypted_update for u in accepted_updates], weights_list,
                )
                t_aggregate = time.time() - t0
                t0 = time.time()
                agg_vec = self.decryptor.decrypt_aggregate(enc_agg).astype(np.float32)
                t_decrypt = time.time() - t0
            elif self.aggregation_method == "fedavg":
                agg_vec = plaintext_aggregate(
                    [u.flat_update for u in accepted_updates], weights_list,
                )
                t_aggregate = time.time() - t0
                t_decrypt = 0.0
            else:
                kwargs = {}
                if self.aggregation_method in ("trimmed_mean",):
                    kwargs["trim_ratio"] = self.trim_ratio
                if self.aggregation_method in ("krum", "multi_krum", "bulyan"):
                    kwargs["num_byzantine"] = self.num_byzantine
                agg_vec = robust_aggregate(
                    self.aggregation_method,
                    [u.flat_update for u in accepted_updates],
                    weights_list,
                    **kwargs,
                )
                t_aggregate = time.time() - t0
                t_decrypt = 0.0
        else:
            self.logger.warning("[%s] round %d: ALL clients rejected; keeping previous model.",
                                self.scenario, round_idx)
            agg_vec = np.zeros_like(self._current_flat_state(), dtype=np.float32)
            t_aggregate = time.time() - t0
            t_decrypt = 0.0

        applied = self._apply_update(agg_vec)
        if applied and accepted_ids:
            self._prev_aggregate = np.asarray(agg_vec, dtype=np.float64).reshape(-1).copy()

        val_metrics = self._evaluate(self.val_loader)

        avg_train_loss = float(np.mean([u.loss_after for u in client_updates]))
        n_malicious_total = sum(1 for u in client_updates if u.is_malicious)
        n_malicious_rejected = sum(
            1 for u in client_updates if u.is_malicious and u.client_id in rejected_ids
        )
        per_client = [{
            "client_id": u.client_id,
            "n_samples": u.n_samples,
            "loss_before": u.loss_before,
            "loss_after": u.loss_after,
            "train_time_sec": u.train_time_sec,
            "is_malicious": u.is_malicious,
            "raw_trust": raw_trust.get(u.client_id, 1.0),
            "smoothed_trust": smoothed.get(u.client_id, 1.0),
            "agg_weight": agg_weights.get(u.client_id, 0.0),
            "accepted": u.client_id in accepted_ids,
        } for u in client_updates]

        return RoundReport(
            round_idx=round_idx,
            scenario=self.scenario,
            aggregation_method=self.aggregation_method,
            use_he=self.use_he,
            use_fuzzy=self.use_fuzzy,
            use_zt=self.use_zt,
            train_loss_avg=avg_train_loss,
            val_loss=val_metrics["loss"],
            val_accuracy=val_metrics["accuracy"],
            val_macro_f1=val_metrics["macro_f1"],
            val_weighted_f1=val_metrics["weighted_f1"],
            accepted_clients=accepted_ids,
            rejected_clients=rejected_ids,
            raw_trust=raw_trust,
            smoothed_trust=smoothed,
            aggregation_weights=agg_weights,
            n_malicious_total=n_malicious_total,
            n_malicious_rejected=n_malicious_rejected,
            time_local_train_sec=t_local,
            time_encrypt_sec=t_enc,
            time_aggregate_sec=t_aggregate,
            time_decrypt_sec=t_decrypt,
            time_fuzzy_sec=t_fuzzy,
            time_total_sec=time.time() - t_round0,
            he_ciphertext_bytes=he_bytes,
            per_client=per_client,
        )

    def evaluate_final(self):
        return self._evaluate(self.test_loader, also_confusion=True)

    def _sample_clients(self) -> List[FederatedClient]:
        frac = float(self.cfg.federated.fraction_fit)
        if frac >= 1.0:
            return list(self.clients)
        k = max(1, int(round(frac * len(self.clients))))
        rng = np.random.default_rng(self.cfg.seed + 1)
        idx = rng.choice(len(self.clients), size=k, replace=False)
        return [self.clients[i] for i in idx]

    def _current_flat_state(self) -> np.ndarray:
        vec, _ = flatten_state(self.model.state_dict())
        return vec

    def _apply_update(self, agg_delta: np.ndarray) -> bool:
        """Apply the aggregated update to the global model.

        Returns ``True`` if the update was applied, ``False`` if the proposed
        new global state contained NaN/Inf and we rolled back to the previous
        round's weights.  Rollback is the conservative thing to do under a
        strong attack because a single NaN-poisoned model is unrecoverable
        (further training compounds the corruption) - in practice it lets
        the experiment continue and lets the ablation show a *flat*
        post-divergence curve instead of crashing entirely.
        """
        global_state = self.model.state_dict()
        flat, schema = flatten_state(global_state)
        if not np.isfinite(agg_delta).all():
            self.logger.warning("[%s] aggregated delta contains NaN/Inf; "
                                "skipping update for this round",
                                self.scenario)
            return False
        new_flat = flat + agg_delta.astype(np.float32)
        if not np.isfinite(new_flat).all():
            self.logger.warning("[%s] new global weights would contain NaN/Inf; "
                                "rolling back to previous round",
                                self.scenario)
            return False
        new_state = unflatten_state(new_flat, schema)
        target_device = next(self.model.parameters()).device
        new_state = {k: v.to(target_device) for k, v in new_state.items()}
        self.model.load_state_dict(new_state)
        return True

    def _evaluate(self, loader: DataLoader, also_confusion: bool = False):
        from ..evaluation.metrics import compute_metrics
        self.model.eval()
        all_y, all_pred, total_loss, n = [], [], 0.0, 0
        loss_fn = nn.CrossEntropyLoss(reduction="sum")
        with torch.no_grad():
            for X, y in loader:
                X = X.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                logits = self.model(X)
                batch_loss = float(loss_fn(logits, y).item())
                if not (batch_loss == batch_loss):  # NaN check
                    batch_loss = float("inf")
                total_loss += batch_loss
                pred = logits.argmax(dim=1)
                all_y.append(y.cpu().numpy())
                all_pred.append(pred.cpu().numpy())
                n += X.size(0)
        y_true = np.concatenate(all_y) if all_y else np.array([])
        y_pred = np.concatenate(all_pred) if all_pred else np.array([])
        mean_loss = total_loss / max(n, 1)
        if not np.isfinite(mean_loss):
            mean_loss = 1.0e6
        metrics = compute_metrics(y_true, y_pred, num_classes=self.num_classes,
                                  loss=mean_loss,
                                  with_confusion=also_confusion)
        return metrics

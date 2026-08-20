"""End-to-end driver for the paper.

Usage
-----
    python run_all.py --config configs/default.yaml
    python run_all.py --config configs/default.yaml --quick   # tiny sanity run
    python run_all.py --scenarios fedavg full_system          # subset of scenarios
    python run_all.py --force-preprocess                      # ignore the cache

Outputs
-------
``results/csv``      Per-round metrics, per-scenario summary, class distribution
                     (one row per round/client; everything the paper needs).
``results/figures``  PNG + PDF figures referenced in the manuscript.
``results/models``   Final model weights for each scenario.
``results/logs``     Full run log (rotated).
"""
from __future__ import annotations

import matplotlib                                              # noqa: E402
matplotlib.use("Agg")                                          # noqa: E402

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data import DataPreprocessor, FederatedPartitioner, TabularDataset, build_loaders
from src.evaluation import (
    compute_metrics, plot_baseline_comparison, plot_class_distribution,
    plot_confusion_matrix, plot_overhead_breakdown, plot_robustness_bars,
    plot_round_curves, plot_trust_heatmap,
)
from src.federated import FederatedClient, FederatedServer, RoundReport
from src.models import build_model
from src.utils import get_logger, load_config, set_global_seed


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FL-IDS + HE + Fuzzy + Zero-Trust")
    p.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    p.add_argument("--scenarios", nargs="+", default=None,
                   help="Subset of scenario names to run (overrides config)")
    p.add_argument("--seeds", nargs="+", type=int, default=None,
                   help="Seeds to run.  Overrides config 'seeds' list.  Each "
                        "scenario is repeated once per seed for std-error bars.")
    p.add_argument("--force-preprocess", action="store_true",
                   help="Rebuild the processed-data cache even if it exists")
    p.add_argument("--dataset", default=None,
                   choices=["cic_iot", "edgeiiot", "keystroke", "rba", "hmog"],
                   help="Dataset to run.  cic_iot (default) uses the merged "
                        "CSV pipeline; the others load pre-processed .npz files "
                        "from the HE-FedSec data_processed folder.")
    p.add_argument("--quick", action="store_true",
                   help="Tiny sanity run: 1 round, 2 clients, 2k samples per class")
    return p.parse_args()


def pick_device(spec: str) -> torch.device:
    if spec == "cpu":
        return torch.device("cpu")
    if spec == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


_LEGACY_MALICIOUS_SCENARIOS = {"full_system"}


def resolve_attack_params(cfg, scen_cfg) -> Dict[str, object]:
    """Pick the attack parameters that apply to this scenario.

    Per-scenario overrides take precedence; missing fields fall through to
    the global ``experiments.malicious_clients`` block.  All scenarios run
    with the same attack within a single experiment so the comparison is
    apples-to-apples - but the design supports per-scenario overrides so a
    follow-up experiment can probe sensitivity.
    """
    base = cfg.experiments.malicious_clients
    attack_type = str(scen_cfg.get("malicious_attack_type", base.attack_type)) \
        if hasattr(scen_cfg, "get") else str(base.attack_type)
    fraction = float(scen_cfg.get("malicious_fraction", base.fraction)) \
        if hasattr(scen_cfg, "get") else float(base.fraction)
    noise_sigma = float(scen_cfg.get("noise_sigma", base.noise_sigma)) \
        if hasattr(scen_cfg, "get") else float(base.noise_sigma)
    return {
        "attack_type": attack_type,
        "fraction": fraction,
        "noise_sigma": noise_sigma,
    }


def maybe_inject_malicious_flags(cfg, num_clients: int, rng: np.random.Generator,
                                 scen_cfg) -> List[bool]:
    """Return a length-``num_clients`` boolean list marking malicious clients.

    The decision comes from the *scenario-level* ``malicious`` flag in the
    config.  Falls back to a name-based whitelist for backwards compatibility
    with older configs.  Setting ``experiments.malicious_clients.enabled = false``
    globally turns every scenario honest (useful for the no-attack sweep).
    """
    if not cfg.experiments.malicious_clients.enabled:
        return [False] * num_clients
    scen_name = scen_cfg.name if hasattr(scen_cfg, "name") else str(scen_cfg)
    if hasattr(scen_cfg, "get"):
        wants_attack = bool(scen_cfg.get("malicious", scen_name in _LEGACY_MALICIOUS_SCENARIOS))
    else:
        wants_attack = scen_name in _LEGACY_MALICIOUS_SCENARIOS
    if not wants_attack:
        return [False] * num_clients
    params = resolve_attack_params(cfg, scen_cfg)
    frac = float(params["fraction"])
    k = int(round(frac * num_clients))
    if k <= 0:
        return [False] * num_clients
    chosen = rng.choice(num_clients, size=k, replace=False)
    flags = [False] * num_clients
    for c in chosen:
        flags[int(c)] = True
    return flags


def apply_quick_overrides(cfg) -> None:
    cfg.data.max_rows_per_class = 2000
    cfg.federated.rounds = 1
    cfg.federated.num_clients = 3
    cfg.federated.local_epochs = 1
    cfg.federated.local_batch_size = 128
    cfg.homomorphic_encryption.key_size = 512
    cfg.seeds = [42]


def resolve_seeds(cfg, cli_seeds: Optional[List[int]]) -> List[int]:
    """Decide which seeds to run: CLI > config list > config 'seed' scalar."""
    if cli_seeds:
        return [int(s) for s in cli_seeds]
    seeds = cfg.get("seeds") if hasattr(cfg, "get") else None
    if seeds:
        return [int(s) for s in seeds]
    return [int(cfg.seed)]


def run_centralized(cfg, split, device: torch.device, logger, seed: int) -> Dict:
    """Centralized baseline: train on the union of all client data.

    Reported as the *upper bound* in the paper.
    """
    set_global_seed(seed)
    in_features = split.num_features
    num_classes = split.num_classes
    model = build_model(cfg, in_features, num_classes).to(device)
    train_loader, val_loader, test_loader = build_loaders(
        split.X_train, split.y_train, split.X_val, split.y_val,
        split.X_test, split.y_test, batch_size=int(cfg.federated.local_batch_size),
    )
    optim = torch.optim.Adam(model.parameters(), lr=float(cfg.federated.client_lr))
    loss_fn = nn.CrossEntropyLoss()
    epochs = int(cfg.federated.rounds)
    history = []

    ms_cfg = cfg.get("model_selection") if hasattr(cfg, "get") else None
    ms_enabled = bool(ms_cfg.get("enabled", True)) if ms_cfg is not None else True
    ms_metric = (str(ms_cfg.get("metric", "macro_f1"))
                 if ms_cfg is not None else "macro_f1")
    ms_higher_is_better = (bool(ms_cfg.get("higher_is_better", True))
                           if ms_cfg is not None else True)
    if ms_metric == "loss" and ms_higher_is_better:
        ms_higher_is_better = False
    best_val = -float("inf") if ms_higher_is_better else float("inf")
    best_state = None
    best_epoch = -1

    for epoch in range(epochs):
        model.train()
        total = 0.0; n = 0
        for X, y in train_loader:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            logits = model(X)
            loss = loss_fn(logits, y)
            loss.backward()
            optim.step()
            total += float(loss.item()) * X.size(0); n += X.size(0)
        model.eval()
        ys, ps, vloss, vn = [], [], 0.0, 0
        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(device); y = y.to(device)
                logits = model(X)
                vloss += float(nn.CrossEntropyLoss(reduction="sum")(logits, y).item())
                ps.append(logits.argmax(1).cpu().numpy())
                ys.append(y.cpu().numpy())
                vn += X.size(0)
        y_true = np.concatenate(ys); y_pred = np.concatenate(ps)
        m = compute_metrics(y_true, y_pred, num_classes=num_classes,
                            loss=vloss / max(vn, 1))
        m.update({"epoch": epoch, "train_loss": total / max(n, 1)})
        history.append(m)
        logger.info("[centralized] epoch %d/%d train_loss=%.4f val_loss=%.4f acc=%.4f f1m=%.4f",
                    epoch + 1, epochs, m["train_loss"], m["loss"],
                    m["accuracy"], m["macro_f1"])

        if ms_enabled:
            cur = float(m.get(ms_metric, m.get("macro_f1", 0.0)))
            improved = (cur > best_val) if ms_higher_is_better else (cur < best_val)
            if improved and np.isfinite(cur):
                best_val = cur
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
                best_epoch = epoch

    if ms_enabled and best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        logger.info("[centralized] restored best-validation checkpoint from "
                    "epoch %d (%s=%.4f) for the test evaluation.",
                    best_epoch + 1, ms_metric, best_val)

    model.eval()
    ys, ps, tloss, tn = [], [], 0.0, 0
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device); y = y.to(device)
            logits = model(X)
            tloss += float(nn.CrossEntropyLoss(reduction="sum")(logits, y).item())
            ps.append(logits.argmax(1).cpu().numpy())
            ys.append(y.cpu().numpy())
            tn += X.size(0)
    y_true = np.concatenate(ys); y_pred = np.concatenate(ps)
    test = compute_metrics(y_true, y_pred, num_classes=num_classes,
                           loss=tloss / max(tn, 1), with_confusion=True)
    return {"history": history, "test": test, "model_state": model.state_dict()}


def build_federated_clients(cfg, split, partition: Dict[int, np.ndarray],
                            model_template: nn.Module, device: torch.device,
                            malicious_flags: List[bool],
                            attack_params: Dict[str, object],
                            proximal_mu: float = 0.0,
                            seed: Optional[int] = None,
                            attack_schedule: Optional[list] = None) -> List[FederatedClient]:
    clients = []
    base_seed = cfg.seed if seed is None else seed
    for cid, idx in partition.items():
        if len(idx) == 0:
            continue
        X = split.X_train[idx]
        y = split.y_train[idx]
        loader = DataLoader(TabularDataset(X, y),
                            batch_size=int(cfg.federated.local_batch_size),
                            shuffle=True, num_workers=0, pin_memory=False,
                            drop_last=False)
        clients.append(FederatedClient(
            client_id=cid,
            model_template=model_template,
            train_loader=loader,
            device=device,
            lr=float(cfg.federated.client_lr),
            local_epochs=int(cfg.federated.local_epochs),
            is_malicious=bool(malicious_flags[cid] if cid < len(malicious_flags) else False),
            malicious_attack=str(attack_params["attack_type"]),
            noise_sigma=float(attack_params["noise_sigma"]),
            num_classes=split.num_classes,
            seed=int(base_seed),
            proximal_mu=float(proximal_mu),
            optimizer=str(cfg.federated.get("optimizer", "sgd"))
                if hasattr(cfg.federated, "get") else "sgd",
            grad_clip_norm=float(cfg.federated.get("grad_clip_norm", 1.0))
                if hasattr(cfg.federated, "get") else 1.0,
            max_update_norm=float(cfg.federated.get("max_update_norm", 100.0))
                if hasattr(cfg.federated, "get") else 100.0,
            attack_schedule=attack_schedule,
        ))
    return clients


def run_scenario(cfg, scen_cfg, split, partition, val_loader, test_loader,
                 device: torch.device, logger, seed: int) -> Dict:
    """Run a single scenario for a given seed.

    ``scen_cfg`` is the per-scenario config block (AttrDict).  Its ``.name``
    field is used both for logging and to back-fill defaults via the legacy
    name->flags table inside the server.
    """
    set_global_seed(seed)  # reset seed for fair comparison across scenarios
    scen_name = str(scen_cfg.name)
    in_features = split.num_features
    num_classes = split.num_classes
    model = build_model(cfg, in_features, num_classes).to(device)
    rng = np.random.default_rng(seed)
    malicious_flags = maybe_inject_malicious_flags(cfg, cfg.federated.num_clients,
                                                   rng, scen_cfg)
    attack_params = resolve_attack_params(cfg, scen_cfg)
    proximal_mu = float(scen_cfg.get("proximal_mu", 0.0)) if hasattr(scen_cfg, "get") else 0.0
    attack_schedule = (list(scen_cfg.get("attack_schedule", []))
                       if hasattr(scen_cfg, "get") else None) or None
    clients = build_federated_clients(cfg, split, partition, model,
                                      device, malicious_flags,
                                      attack_params=attack_params,
                                      proximal_mu=proximal_mu,
                                      seed=seed,
                                      attack_schedule=attack_schedule)
    server = FederatedServer(cfg, scen_cfg, model, clients, val_loader,
                             test_loader, num_classes, device,
                             logger=logger, project_root=PROJECT_ROOT)
    reports = server.run()
    test = server.evaluate_final()
    return {
        "scenario": scen_name,
        "scenario_cfg": scen_cfg,
        "seed": int(seed),
        "attack_params": attack_params,
        "reports": reports,
        "test": test,
        "malicious_flags": malicious_flags,
        "final_state": model.state_dict(),
    }


def write_round_csv(all_reports: List, out_csv: Path) -> pd.DataFrame:
    """Persist per-(scenario, seed, round) metrics.

    ``all_reports`` may hold either raw :class:`RoundReport` instances or
    ``(seed, report)`` tuples - both forms are accepted so callers do not
    need to flatten themselves.
    """
    rows = []
    for item in all_reports:
        if isinstance(item, tuple):
            seed, r = item
        else:
            seed, r = getattr(item, "seed", 0), item
        rows.append({
            "seed": int(seed),
            "scenario": r.scenario,
            "aggregation_method": getattr(r, "aggregation_method", "fedavg"),
            "use_he": bool(getattr(r, "use_he", False)),
            "use_fuzzy": bool(getattr(r, "use_fuzzy", False)),
            "use_zt": bool(getattr(r, "use_zt", False)),
            "round_idx": r.round_idx,
            "train_loss_avg": r.train_loss_avg,
            "val_loss": r.val_loss,
            "val_accuracy": r.val_accuracy,
            "val_macro_f1": r.val_macro_f1,
            "val_weighted_f1": r.val_weighted_f1,
            "n_accepted": len(r.accepted_clients),
            "n_rejected": len(r.rejected_clients),
            "n_malicious_total": r.n_malicious_total,
            "n_malicious_rejected": r.n_malicious_rejected,
            "time_local_train_sec": r.time_local_train_sec,
            "time_encrypt_sec": r.time_encrypt_sec,
            "time_aggregate_sec": r.time_aggregate_sec,
            "time_decrypt_sec": r.time_decrypt_sec,
            "time_fuzzy_sec": r.time_fuzzy_sec,
            "time_total_sec": r.time_total_sec,
            "he_ciphertext_bytes": r.he_ciphertext_bytes,
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df


def write_per_client_csv(all_reports: List, out_csv: Path) -> pd.DataFrame:
    rows = []
    for item in all_reports:
        if isinstance(item, tuple):
            seed, r = item
        else:
            seed, r = getattr(item, "seed", 0), item
        for entry in r.per_client:
            rows.append({"seed": int(seed), "scenario": r.scenario,
                         "round_idx": r.round_idx, **entry})
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df


def write_per_seed_summary_csv(centralized_results: List[Dict],
                               fl_results: List[Dict],
                               out_csv: Path) -> pd.DataFrame:
    """One row per (scenario, seed)."""
    rows = []
    for r in centralized_results or []:
        t = r["test"]
        rows.append({"scenario": "centralized", "seed": int(r["seed"]),
                     "test_loss": t.get("loss", 0.0),
                     "test_accuracy": t["accuracy"],
                     "test_macro_f1": t["macro_f1"],
                     "test_weighted_f1": t["weighted_f1"]})
    for r in fl_results:
        t = r["test"]
        rows.append({"scenario": r["scenario"], "seed": int(r["seed"]),
                     "test_loss": t.get("loss", 0.0),
                     "test_accuracy": t["accuracy"],
                     "test_macro_f1": t["macro_f1"],
                     "test_weighted_f1": t["weighted_f1"]})
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df


def write_summary_csv(centralized_results: List[Dict], fl_results: List[Dict],
                      out_csv: Path) -> pd.DataFrame:
    """Aggregate test metrics across seeds: mean +/- std per scenario."""
    per_seed = write_per_seed_summary_csv(centralized_results, fl_results,
                                          out_csv.with_name("scenario_per_seed.csv"))
    if per_seed.empty:
        per_seed.to_csv(out_csv, index=False)
        return per_seed
    agg_cols = ["test_loss", "test_accuracy", "test_macro_f1", "test_weighted_f1"]
    g = per_seed.groupby("scenario")[agg_cols]
    summary = g.agg(["mean", "std", "count"]).reset_index()
    summary.columns = [
        "scenario" if a == "scenario" else f"{a}_{b}"
        for a, b in summary.columns
    ]
    for c in agg_cols:
        summary[c] = summary[f"{c}_mean"]
    summary.to_csv(out_csv, index=False)
    return summary


def write_trust_heatmap_csv(fl_results: List[Dict], out_dir: Path) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for r in fl_results:
        scen = r["scenario"]
        if scen not in ("fedavg_he_fuzzy", "full_system"):
            continue
        round_idxs = sorted({rep.round_idx for rep in r["reports"]})
        client_ids = sorted({cid for rep in r["reports"] for cid in rep.raw_trust})
        mat = pd.DataFrame(index=client_ids, columns=round_idxs, dtype=float)
        for rep in r["reports"]:
            for cid, score in rep.raw_trust.items():
                mat.loc[cid, rep.round_idx] = float(score)
        mat = mat.fillna(0.0)
        out_csv = out_dir / f"trust_matrix_{scen}.csv"
        mat.to_csv(out_csv)
        out[scen] = mat
    return out


def run_experiment(cfg, scenarios_filter: Optional[List[str]] = None,
                   seeds_override: Optional[List[int]] = None,
                   force_preprocess: bool = False,
                   logger_name: str = "run_all") -> Dict:
    """Run one experiment (all scenarios x all seeds) using ``cfg``.

    Used by both :func:`main` and the ablation runner so the exact same
    pipeline is exercised either way.  Returns a dict with the aggregated
    summary and per-round DataFrames plus all paths it wrote to.
    """
    set_global_seed(cfg.seed)
    device = pick_device(cfg.device)

    results_dir = PROJECT_ROOT / cfg.paths.results_dir
    figures_dir = PROJECT_ROOT / cfg.paths.figures_dir
    csv_dir = PROJECT_ROOT / cfg.paths.csv_dir
    models_dir = PROJECT_ROOT / cfg.paths.models_dir
    logs_dir = PROJECT_ROOT / cfg.paths.logs_dir
    for d in (results_dir, figures_dir, csv_dir, models_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)
    logger = get_logger(logger_name, logs_dir, level=cfg.logging.level)
    logger.info("=" * 72)
    logger.info("Starting experiment | device=%s | seed=%d", device, cfg.seed)
    logger.info("Results dir: %s", results_dir)

    t0 = time.time()
    data_source = str(cfg.data.get("source", "cic_iot")) if hasattr(cfg.data, "get") else "cic_iot"
    if data_source == "npz":
        from src.data.npz_preprocessor import NpzPreprocessor
        prep = NpzPreprocessor(cfg, PROJECT_ROOT)
    elif data_source == "hmog":
        from src.data.hmog_preprocessor import HmogPreprocessor
        prep = HmogPreprocessor(cfg, PROJECT_ROOT)
    else:
        prep = DataPreprocessor(cfg, PROJECT_ROOT)
    split = prep.run(force=force_preprocess)
    natural_pkey = getattr(prep, "partition_key_", None)
    logger.info("Dataset ready: train=%d val=%d test=%d features=%d classes=%d (%.1fs)",
                len(split.y_train), len(split.y_val), len(split.y_test),
                split.num_features, split.num_classes, time.time() - t0)
    logger.info("Label set: %s", split.label_names)

    dist_csv = csv_dir / "class_distribution_after_subsample.csv"
    if dist_csv.exists():
        plot_class_distribution(dist_csv, figures_dir / "class_distribution",
                                title="CIC-IoT-2023 class distribution after subsampling")

    partitioner = FederatedPartitioner(
        num_clients=int(cfg.federated.num_clients),
        strategy=str(cfg.federated.partition),
        alpha=float(cfg.federated.dirichlet_alpha),
        seed=cfg.seed,
        logger=logger,
    )
    partition = partitioner.split(split.y_train)
    part_rows = []
    for cid, idx in partition.items():
        counts = np.bincount(split.y_train[idx], minlength=split.num_classes)
        row = {"client_id": cid, "n_samples": len(idx)}
        for c in range(split.num_classes):
            row[f"count[{split.label_names[c]}]"] = int(counts[c])
        part_rows.append(row)
    pd.DataFrame(part_rows).to_csv(csv_dir / "federated_partition.csv", index=False)

    val_loader_global = DataLoader(TabularDataset(split.X_val, split.y_val),
                                   batch_size=int(cfg.federated.local_batch_size) * 4,
                                   shuffle=False)
    test_loader_global = DataLoader(TabularDataset(split.X_test, split.y_test),
                                    batch_size=int(cfg.federated.local_batch_size) * 4,
                                    shuffle=False)

    scen_by_name = {s.name: s for s in cfg.experiments.scenarios}
    all_names = list(scen_by_name.keys())
    if scenarios_filter:
        wanted = [s for s in scenarios_filter if s in all_names + ["centralized"]]
    else:
        wanted = list(all_names)
    seeds = resolve_seeds(cfg, seeds_override)
    logger.info("Running scenarios: %s", wanted)
    logger.info("Seeds: %s", seeds)

    centralized_results: List[Dict] = []
    fl_results: List[Dict] = []
    all_reports_with_seed: List[tuple] = []

    for seed in seeds:
        logger.info("================================================================")
        logger.info("SEED %d", seed)
        logger.info("================================================================")
        partitioner = FederatedPartitioner(
            num_clients=int(cfg.federated.num_clients),
            strategy=str(cfg.federated.partition),
            alpha=float(cfg.federated.dirichlet_alpha),
            seed=int(seed),
            logger=logger,
        )
        partition = partitioner.split(split.y_train)
        if seed == seeds[0]:
            part_rows = []
            for cid, idx in partition.items():
                counts = np.bincount(split.y_train[idx], minlength=split.num_classes)
                row = {"seed": int(seed), "client_id": cid, "n_samples": len(idx)}
                for c in range(split.num_classes):
                    row[f"count[{split.label_names[c]}]"] = int(counts[c])
                part_rows.append(row)
            pd.DataFrame(part_rows).to_csv(csv_dir / "federated_partition.csv", index=False)

        for scen_name in wanted:
            scen_t0 = time.time()
            if scen_name == "centralized":
                logger.info("--- [seed=%d] Centralized baseline ---", seed)
                cr = run_centralized(cfg, split, device, logger, seed)
                cr["seed"] = int(seed)
                centralized_results.append(cr)
                torch.save(cr["model_state"], models_dir / f"centralized_seed{seed}.pt")
                hist_df = pd.DataFrame(cr["history"])
                hist_df.insert(0, "seed", int(seed))
                hist_df.to_csv(csv_dir / f"centralized_history_seed{seed}.csv", index=False)
                cm = cr["test"].get("confusion_matrix")
                if cm is not None and seed == seeds[0]:
                    plot_confusion_matrix(cm, split.label_names,
                                          figures_dir / "confusion_centralized",
                                          title="Centralized - confusion matrix (test)")
                logger.info("[centralized seed=%d] done in %.1fs", seed, time.time() - scen_t0)
                continue

            if scen_name not in scen_by_name:
                logger.warning("Unknown scenario: %s", scen_name)
                continue
            scen_cfg = scen_by_name[scen_name]
            logger.info("--- [seed=%d] Scenario: %s | %s ---", seed, scen_name,
                        scen_cfg.get("description", ""))
            result = run_scenario(cfg, scen_cfg, split, partition, val_loader_global,
                                  test_loader_global, device, logger, seed=seed)
            torch.save(result["final_state"],
                       models_dir / f"{scen_name}_seed{seed}.pt")
            for rep in result["reports"]:
                all_reports_with_seed.append((seed, rep))
            fl_results.append(result)

            cm = result["test"].get("confusion_matrix")
            if cm is not None and seed == seeds[0]:
                plot_confusion_matrix(cm, split.label_names,
                                      figures_dir / f"confusion_{scen_name}",
                                      title=f"{scen_name} - confusion matrix (test)")
            logger.info("[%s seed=%d] done in %.1fs | test_acc=%.4f f1m=%.4f",
                        scen_name, seed, time.time() - scen_t0,
                        result["test"]["accuracy"], result["test"]["macro_f1"])

    per_round_df = pd.DataFrame()
    if all_reports_with_seed:
        per_round_df = write_round_csv(all_reports_with_seed,
                                       csv_dir / "per_round_metrics.csv")
        write_per_client_csv(all_reports_with_seed,
                             csv_dir / "per_client_round_metrics.csv")
        plot_round_curves(per_round_df, figures_dir / "convergence_curves")
        plot_overhead_breakdown(per_round_df, figures_dir / "overhead_breakdown")

    summary_df = write_summary_csv(centralized_results, fl_results,
                                   csv_dir / "scenario_summary.csv")
    plot_robustness_bars(summary_df, figures_dir / "scenario_summary_bars")

    if not per_round_df.empty:
        baseline_scenarios = [
            n for n in ("fedavg_attack", "fedprox", "fedmedian",
                        "trimmed_mean", "krum")
            if n in summary_df["scenario"].values
        ]
        proposed_scenarios = [
            n for n in ("fedavg_he", "fedavg_he_fuzzy", "full_system")
            if n in summary_df["scenario"].values
        ]
        if baseline_scenarios and proposed_scenarios:
            plot_baseline_comparison(per_round_df, summary_df,
                                     figures_dir / "baseline_vs_proposed",
                                     baseline_scenarios=baseline_scenarios,
                                     proposed_scenarios=proposed_scenarios)

    seen = set()
    first_seed_results = []
    for r in fl_results:
        if r["scenario"] not in seen:
            seen.add(r["scenario"])
            first_seed_results.append(r)
    trust_mats = write_trust_heatmap_csv(first_seed_results, csv_dir)
    for scen, mat in trust_mats.items():
        matches = [r for r in first_seed_results if r["scenario"] == scen]
        if not matches:
            continue
        mal_ids = [cid for cid, flag in enumerate(matches[0]["malicious_flags"]) if flag]
        plot_trust_heatmap(mat, figures_dir / f"trust_heatmap_{scen}",
                           title=f"Per-round raw trust score - {scen}",
                           malicious_ids=mal_ids)

    dump = {
        "config": cfg.to_plain(),
        "seeds": seeds,
        "label_names": split.label_names,
        "feature_names": split.feature_names,
        "summary": summary_df.to_dict(orient="records"),
    }
    (results_dir / "run_summary.json").write_text(
        json.dumps(dump, indent=2, default=str), encoding="utf-8"
    )

    logger.info("All done. Results under %s", results_dir)
    return {
        "summary": summary_df,
        "per_round": per_round_df,
        "results_dir": results_dir,
        "centralized_results": centralized_results,
        "fl_results": fl_results,
    }


NPZ_DATASETS = {
    "edgeiiot":  "D:/ChiVan/Dataset/HE-FedSec/data_processed/edgeiiot.npz",
    "keystroke": "D:/ChiVan/Dataset/HE-FedSec/data_processed/keystroke.npz",
    "rba":       "D:/ChiVan/Dataset/HE-FedSec/data_processed/rba.npz",
}


def apply_dataset(cfg, dataset: str) -> None:
    """Reconfigure ``cfg`` in place to run on the requested dataset.

    For the .npz datasets we switch the data source, point at the file, and
    redirect every results subdir to ``results_<dataset>/`` so multi-dataset
    runs never clobber the primary CIC-IoT results.
    """
    if not dataset or dataset == "cic_iot":
        return
    if dataset == "hmog":
        cfg.data["source"] = "hmog"
        cfg.model["type"] = "cnn_lstm"
        cfg.model["input_shape"] = [128, 6]
        cfg.federated["optimizer"] = "adam"
        cfg.federated["client_lr"] = 0.002
        cfg.federated["local_epochs"] = 2
    else:
        cfg.data["source"] = "npz"
        cfg.data["npz_path"] = NPZ_DATASETS[dataset]
        cfg.data["npz_name"] = dataset
        cfg.federated["optimizer"] = "adam"
        cfg.federated["client_lr"] = 0.003
        cfg.federated["local_epochs"] = 2
    base = f"results_{dataset}"
    cfg.paths["results_dir"] = base
    cfg.paths["figures_dir"] = f"{base}/figures"
    cfg.paths["csv_dir"] = f"{base}/csv"
    cfg.paths["models_dir"] = f"{base}/models"
    cfg.paths["logs_dir"] = f"{base}/logs"


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.dataset:
        apply_dataset(cfg, args.dataset)
    if args.quick:
        apply_quick_overrides(cfg)
    return run_experiment(
        cfg,
        scenarios_filter=args.scenarios,
        seeds_override=args.seeds,
        force_preprocess=args.force_preprocess,
    )


if __name__ == "__main__":
    main()

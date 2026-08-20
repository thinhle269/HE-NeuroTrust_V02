"""Case Study: Adaptive vs. Static Hyperparameters under a Time-Varying Threat.

Motivation
----------
The main paper picks a single operating point (tau = 0.20, ema_alpha = 0.20)
optimised on a stationary attack profile.  In a real deployment the
threat profile *changes* over time:

* the attacker stays quiet during the warm-up phase (no attack),
* then turns on a sign-flip campaign,
* then notices that the defence has adapted and switches to label-flip.

A static threshold cannot follow this trajectory: it is either too lenient
for the active-attack phase or too strict for the warm-up.  We therefore
run *three* variants of `full_system` on the *same* time-varying threat
and compare their convergence:

* ``static_tau020``  - the manuscript's main default (tau=0.20)
* ``static_tau040``  - a conservative static choice
* ``adaptive``       - both controllers from src/zerotrust/adaptive.py
                       react in real time to val_macro_f1 and the
                       per-client trust drift.

Outputs (under ``results/case_study/``):
    csv/per_round_metrics.csv     - all three variants joined
    csv/adaptive_trajectory.csv   - the controller's tau/alpha per round
    figures/case_study.{png,pdf}  - 3-panel comparison
    figures/adaptive_trajectory.{png,pdf}
                                  - tau and alpha vs round overlaid on
                                    the attack schedule
"""
from __future__ import annotations

import matplotlib                                              # noqa: E402
matplotlib.use("Agg")                                          # noqa: E402

import copy
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader

from src.data import DataPreprocessor, FederatedPartitioner, TabularDataset
from src.utils import get_logger, load_config, set_global_seed
from src.models import build_model
from src.federated import FederatedClient, FederatedServer

PROJECT_ROOT = Path(__file__).resolve().parent
CASE_ROOT = PROJECT_ROOT / "results" / "case_study"


ATTACK_SCHEDULE = [
    (0, "no_attack"),    # warm-up - act honest
    (10, "sign_flip"),   # switch on sign-flip
    (20, "label_flip"),  # adapt: switch to label-flip
]


VARIANTS = {
    "static_tau020": {
        "trust_threshold": 0.20,
        "ema_alpha": 0.20,
        "adaptive": None,
    },
    "static_tau040": {
        "trust_threshold": 0.40,
        "ema_alpha": 0.20,
        "adaptive": None,
    },
    "adaptive": {
        "trust_threshold": 0.20,     # initial value; controller will update
        "ema_alpha": 0.20,           # initial value; controller will update
        "adaptive": {
            "tau":   {"enabled": True,
                      "init": 0.20, "min": 0.10, "max": 0.45,
                      "f1_window": 3, "k_up": 0.05, "k_down": 0.02,
                      "drop_trigger": 0.02, "recover_trigger": 0.02},
            "alpha": {"enabled": True,
                      "init": 0.20, "min": 0.20, "max": 0.70,
                      "change_low": 0.05, "change_high": 0.20},
        },
    },
}


def make_case_study_cfg(base_cfg, variant_name: str) -> object:
    """Deep-copy the default config and patch in the variant's knobs."""
    cfg = copy.deepcopy(base_cfg)
    v = VARIANTS[variant_name]
    cfg.zero_trust.trust_threshold = float(v["trust_threshold"])
    cfg.zero_trust["ema_alpha"] = float(v["ema_alpha"])
    if v["adaptive"] is None:
        cfg.zero_trust["adaptive"] = None
    else:
        cfg.zero_trust["adaptive"] = v["adaptive"]
    cfg.seeds = [42]
    cfg.federated.rounds = 30
    cfg.data.max_rows_per_class = 8000
    cfg.paths["processed_dir"] = "data/processed_case_study"
    cfg.paths["results_dir"] = f"results/case_study/{variant_name}"
    cfg.paths["figures_dir"] = f"results/case_study/{variant_name}/figures"
    cfg.paths["csv_dir"] = f"results/case_study/{variant_name}/csv"
    cfg.paths["models_dir"] = f"results/case_study/{variant_name}/models"
    cfg.paths["logs_dir"] = f"results/case_study/{variant_name}/logs"
    return cfg


def run_single_variant(variant_name: str, base_cfg, split, partition,
                       val_loader, test_loader, device, logger) -> Dict:
    """Run one variant against the time-varying attack schedule."""
    cfg = make_case_study_cfg(base_cfg, variant_name)
    set_global_seed(cfg.seed)
    model = build_model(cfg, split.num_features, split.num_classes).to(device)

    rng = np.random.default_rng(cfg.seed)
    n_clients = int(cfg.federated.num_clients)
    n_mal = max(1, int(round(0.30 * n_clients)))
    mal_idx = set(rng.choice(n_clients, size=n_mal, replace=False).tolist())
    malicious_flags = [i in mal_idx for i in range(n_clients)]
    logger.info("[%s] malicious clients: %s", variant_name,
                sorted(mal_idx))

    clients = []
    for cid, idx in partition.items():
        if len(idx) == 0:
            continue
        loader = DataLoader(TabularDataset(split.X_train[idx], split.y_train[idx]),
                            batch_size=int(cfg.federated.local_batch_size),
                            shuffle=True, num_workers=0)
        clients.append(FederatedClient(
            client_id=cid,
            model_template=model,
            train_loader=loader,
            device=device,
            lr=float(cfg.federated.client_lr),
            local_epochs=int(cfg.federated.local_epochs),
            is_malicious=bool(malicious_flags[cid]),
            malicious_attack=str(cfg.experiments.malicious_clients.attack_type),
            noise_sigma=float(cfg.experiments.malicious_clients.noise_sigma),
            num_classes=split.num_classes,
            seed=int(cfg.seed),
            proximal_mu=0.0,
            grad_clip_norm=float(cfg.federated.get("grad_clip_norm", 1.0)),
            max_update_norm=float(cfg.federated.get("max_update_norm", 100.0)),
            attack_schedule=ATTACK_SCHEDULE,
        ))

    scen_cfg = type("S", (), {})()
    scen_cfg.name = variant_name
    scen_cfg.get = lambda k, default=None: {
        "name": variant_name,
        "aggregation": "fedavg",
        "he": True,
        "fuzzy": True,
        "zero_trust": True,
        "malicious": True,
    }.get(k, default)
    server = FederatedServer(cfg, scen_cfg, model, clients, val_loader,
                             test_loader, split.num_classes, device,
                             logger=logger, project_root=PROJECT_ROOT)
    t0 = time.time()
    reports = server.run()
    test = server.evaluate_final()
    elapsed = time.time() - t0
    logger.info("[%s] done in %.1fs | test_acc=%.4f f1m=%.4f",
                variant_name, elapsed, test["accuracy"], test["macro_f1"])

    rows = []
    for r in reports:
        rows.append({
            "variant": variant_name,
            "round_idx": r.round_idx,
            "val_loss": r.val_loss,
            "val_accuracy": r.val_accuracy,
            "val_macro_f1": r.val_macro_f1,
            "n_accepted": len(r.accepted_clients),
            "n_rejected": len(r.rejected_clients),
            "n_malicious_total": r.n_malicious_total,
            "n_malicious_rejected": r.n_malicious_rejected,
            "tau_effective": float(getattr(server.zt, "threshold", float("nan"))),
            "ema_alpha_effective": float(getattr(server.zt, "ema_alpha", float("nan"))),
        })
    per_round_df = pd.DataFrame(rows)

    traj = pd.DataFrame()
    if server.adaptive:
        records = []
        for k, ctrl in server.adaptive.items():
            for entry in ctrl.history:
                records.append({"variant": variant_name, "controller": k, **entry})
        traj = pd.DataFrame(records)

    return {
        "variant": variant_name,
        "test": test,
        "per_round": per_round_df,
        "trajectory": traj,
    }


def main():
    base = load_config(PROJECT_ROOT / "configs" / "default.yaml")
    set_global_seed(base.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CASE_ROOT.mkdir(parents=True, exist_ok=True)
    (CASE_ROOT / "logs").mkdir(parents=True, exist_ok=True)
    (CASE_ROOT / "csv").mkdir(parents=True, exist_ok=True)
    (CASE_ROOT / "figures").mkdir(parents=True, exist_ok=True)
    logger = get_logger("case_study", CASE_ROOT / "logs",
                        level=base.logging.level)

    logger.info("=" * 72)
    logger.info("Adaptive vs. Static Case Study | device=%s", device)
    logger.info("Attack schedule: %s", ATTACK_SCHEDULE)

    case_cfg = make_case_study_cfg(base, "static_tau020")
    prep = DataPreprocessor(case_cfg, PROJECT_ROOT)
    split = prep.run(force=False)
    logger.info("Dataset ready: train=%d val=%d test=%d",
                len(split.y_train), len(split.y_val), len(split.y_test))

    partitioner = FederatedPartitioner(
        num_clients=int(case_cfg.federated.num_clients),
        strategy=str(case_cfg.federated.partition),
        alpha=float(case_cfg.federated.dirichlet_alpha),
        seed=int(case_cfg.seed),
        logger=logger,
    )
    partition = partitioner.split(split.y_train)
    val_loader = DataLoader(TabularDataset(split.X_val, split.y_val),
                            batch_size=int(case_cfg.federated.local_batch_size) * 4,
                            shuffle=False)
    test_loader = DataLoader(TabularDataset(split.X_test, split.y_test),
                             batch_size=int(case_cfg.federated.local_batch_size) * 4,
                             shuffle=False)

    all_rounds: List[pd.DataFrame] = []
    all_traj: List[pd.DataFrame] = []
    summary_rows = []
    for variant in VARIANTS:
        logger.info("--- variant: %s ---", variant)
        try:
            res = run_single_variant(variant, base, split, partition,
                                     val_loader, test_loader, device, logger)
            all_rounds.append(res["per_round"])
            if not res["trajectory"].empty:
                all_traj.append(res["trajectory"])
            summary_rows.append({
                "variant": variant,
                "test_accuracy": res["test"]["accuracy"],
                "test_macro_f1": res["test"]["macro_f1"],
                "test_weighted_f1": res["test"]["weighted_f1"],
            })
        except Exception as exc:
            logger.exception("variant %s crashed: %s", variant, exc)

    if all_rounds:
        rounds_df = pd.concat(all_rounds, ignore_index=True)
        rounds_df.to_csv(CASE_ROOT / "csv" / "per_round_metrics.csv", index=False)
    if all_traj:
        traj_df = pd.concat(all_traj, ignore_index=True)
        traj_df.to_csv(CASE_ROOT / "csv" / "adaptive_trajectory.csv", index=False)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(CASE_ROOT / "csv" / "summary.csv", index=False)
    logger.info("Summary: \n%s", summary_df.to_string(index=False))

    if all_rounds:
        render_figures(rounds_df, all_traj, CASE_ROOT / "figures")
    logger.info("Case study done. Outputs under %s", CASE_ROOT)


def render_figures(rounds_df: pd.DataFrame, traj_dfs: List[pd.DataFrame],
                   out_dir: Path) -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    palette = {"static_tau020": "#1f77b4",
               "static_tau040": "#ff7f0e",
               "adaptive": "#2ca02c"}
    for variant, sub in rounds_df.groupby("variant"):
        sub = sub.sort_values("round_idx")
        x = sub["round_idx"] + 1
        axes[0].plot(x, sub["val_macro_f1"], marker="o", lw=2,
                     label=variant, color=palette.get(variant, "gray"))
        axes[1].plot(x, sub["n_malicious_rejected"], marker="o", lw=2,
                     label=variant, color=palette.get(variant, "gray"))
        axes[2].plot(x, sub["n_rejected"], marker="o", lw=2,
                     label=variant, color=palette.get(variant, "gray"))
    for ax in axes:
        ax.axvspan(0.5, 10.5, alpha=0.08, color="gray", label="warm-up" if ax is axes[0] else None)
        ax.axvspan(10.5, 20.5, alpha=0.10, color="red", label="sign-flip" if ax is axes[0] else None)
        ax.axvspan(20.5, 30.5, alpha=0.10, color="purple", label="label-flip" if ax is axes[0] else None)
    axes[0].set_title("Val macro-F1 (higher = better)")
    axes[1].set_title("Malicious clients rejected (out of 3)")
    axes[2].set_title("Total rejected per round (lower honest cost = better)")
    for ax in axes:
        ax.set_xlabel("Round")
    axes[0].legend(loc="lower right", fontsize=8, ncol=1)
    fig.savefig(out_dir / "case_study.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "case_study.pdf", bbox_inches="tight")
    plt.close(fig)

    if traj_dfs:
        traj = pd.concat(traj_dfs, ignore_index=True)
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        for ctrl, sub in traj.groupby("controller"):
            sub = sub.sort_values("round_idx")
            ax = axes[0] if ctrl == "tau" else axes[1]
            value_col = "tau_next" if ctrl == "tau" else "alpha_next"
            ax.plot(sub["round_idx"] + 1, sub[value_col], marker="o", lw=2,
                    color="#2ca02c")
            ax.set_ylabel(f"{ctrl} (effective)")
        for ax in axes:
            ax.axvspan(0.5, 10.5, alpha=0.08, color="gray")
            ax.axvspan(10.5, 20.5, alpha=0.10, color="red")
            ax.axvspan(20.5, 30.5, alpha=0.10, color="purple")
        axes[0].axhline(0.20, color="gray", ls=":", lw=1.0, label="static tau=0.20")
        axes[1].axhline(0.20, color="gray", ls=":", lw=1.0, label="static alpha=0.20")
        axes[0].legend(); axes[1].legend()
        axes[1].set_xlabel("Round")
        axes[0].set_title("Adaptive controllers: trajectory over time")
        fig.savefig(out_dir / "adaptive_trajectory.png", dpi=300, bbox_inches="tight")
        fig.savefig(out_dir / "adaptive_trajectory.pdf", bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()

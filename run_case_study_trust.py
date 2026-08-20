"""Case study: trust dynamics of Byzantine rejection under HE-NeuroTrust.

Runs the full HE-NeuroTrust (neuro-fuzzy) scenario on CIC-IoT-2023 (seed 42,
30% sign-flip) and records, for every client at every round, the smoothed
Zero-Trust score, the aggregation weight, and the accept/reject decision.  It
then plots how the neuro-fuzzy engine + Zero-Trust policy drive the malicious
clients below the acceptance gate (tau = 0.40) and zero their aggregation
weight, while honest clients stay trusted - a concrete walk-through of the
mechanism in Section 3.5.

Outputs:
  results/case_study/csv/trust_trajectory.csv      - per client x round record
  new_paper/figures/fig_case_study_trust.{png,pdf} - two-panel trajectory

Run:  python run_case_study_trust.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.utils import load_config, get_logger, set_global_seed
from src.data import DataPreprocessor, FederatedPartitioner, TabularDataset
import run_all

FIG = ROOT / "new_paper" / "figures"; FIG.mkdir(parents=True, exist_ok=True)
CS = ROOT / "results" / "case_study" / "csv"; CS.mkdir(parents=True, exist_ok=True)
TAU = 0.40
SCEN = "full_system_neuro"
SEED = 42


def run():
    cfg = load_config(ROOT / "configs/default.yaml")
    logger = get_logger("case_study_trust", ROOT / "results/case_study/logs",
                        level=cfg.logging.level)
    set_global_seed(SEED)
    device = torch.device("cpu")

    prep = DataPreprocessor(cfg, ROOT)
    split = prep.run(force=False)
    partitioner = FederatedPartitioner(
        num_clients=int(cfg.federated.num_clients),
        strategy=str(cfg.federated.partition),
        alpha=float(cfg.federated.dirichlet_alpha), seed=SEED, logger=logger)
    partition = partitioner.split(split.y_train)
    val_loader = DataLoader(TabularDataset(split.X_val, split.y_val),
                            batch_size=int(cfg.federated.local_batch_size) * 4, shuffle=False)
    test_loader = DataLoader(TabularDataset(split.X_test, split.y_test),
                             batch_size=int(cfg.federated.local_batch_size) * 4, shuffle=False)

    scen_by_name = {s.name: s for s in cfg.experiments.scenarios}
    scen_cfg = scen_by_name[SCEN]
    logger.info("Running case-study scenario %s (seed %d)", SCEN, SEED)
    res = run_all.run_scenario(cfg, scen_cfg, split, partition, val_loader,
                               test_loader, device, logger, seed=SEED)

    rows = []
    for rep in res["reports"]:
        for pc in rep.per_client:
            rows.append({"round": rep.round_idx, **pc})
    df = pd.DataFrame(rows)
    df.to_csv(CS / "trust_trajectory.csv", index=False)
    logger.info("Wrote %s (%d rows)", CS / "trust_trajectory.csv", len(df))
    return df


def plot(df: pd.DataFrame):
    df = df.copy(); df["round"] = df["round"] + 1  # 1-indexed for display
    mal_ids = sorted(df[df.is_malicious].client_id.unique().tolist())
    hon_ids = sorted(df[~df.is_malicious].client_id.unique().tolist())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.6))

    for cid in hon_ids:
        s = df[df.client_id == cid].sort_values("round")
        ax1.plot(s["round"], s["smoothed_trust"], color="#2ca02c", lw=1.3, alpha=0.55)
    for cid in mal_ids:
        s = df[df.client_id == cid].sort_values("round")
        ax1.plot(s["round"], s["smoothed_trust"], color="#d62728", lw=2.0, marker="o", ms=3)
    ax1.axhline(TAU, color="#1f3864", lw=1.6, ls="--")
    ax1.text(df["round"].max() * 0.98, TAU + 0.02, "Zero-Trust gate  $\\tau=0.40$",
             ha="right", fontsize=9, color="#1f3864", weight="bold")
    ax1.plot([], [], color="#2ca02c", lw=1.8, label=f"honest clients (n={len(hon_ids)})")
    ax1.plot([], [], color="#d62728", lw=2.0, marker="o", ms=3,
             label=f"malicious clients (n={len(mal_ids)})")
    ax1.set_xlabel("federated round"); ax1.set_ylabel("smoothed Zero-Trust score")
    ax1.set_ylim(0, 1.02)
    ax1.set_title("(a) Trust trajectories: malicious clients driven below the gate",
                  fontsize=10, weight="bold")
    ax1.legend(fontsize=8.5, loc="center right")
    ax1.grid(True, alpha=0.3)

    g = df.groupby(["round", "is_malicious"]).agg(
        w=("agg_weight", "mean"), acc=("accepted", "mean")).reset_index()
    hon = g[~g.is_malicious].sort_values("round")
    mal = g[g.is_malicious].sort_values("round")
    ax2.plot(hon["round"], hon["w"], color="#2ca02c", lw=2.0, marker="o", ms=3,
             label="honest (mean aggregation weight)")
    ax2.plot(mal["round"], mal["w"], color="#d62728", lw=2.0, marker="s", ms=3,
             label="malicious (mean aggregation weight)")
    ax2.axhline(0, color="gray", lw=0.8)
    ax2.set_xlabel("federated round"); ax2.set_ylabel("mean aggregation weight $\\alpha_i$")
    ax2.set_title("(b) Malicious clients receive (near-)zero aggregation weight",
                  fontsize=10, weight="bold")
    ax2.legend(fontsize=8.5, loc="center right")
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Case study - HE-NeuroTrust neutralising a 30% Byzantine minority "
                 "(CIC-IoT-2023, seed 42)", fontsize=11, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIG / "fig_case_study_trust.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / "fig_case_study_trust.pdf", bbox_inches="tight")
    plt.close(fig)

    post = df[df["round"] >= 6]  # after neuro warm-up
    mh = post[post.is_malicious].smoothed_trust.mean()
    hh = post[~post.is_malicious].smoothed_trust.mean()
    rej = post[post.is_malicious].accepted.mean()
    print(f"[case study] post-warmup mean trust: malicious={mh:.3f} honest={hh:.3f}")
    print(f"[case study] malicious acceptance rate post-warmup = {rej*100:.1f}% "
          f"(rejection = {(1-rej)*100:.1f}%)")
    print("fig_case_study_trust ok ->", FIG)


if __name__ == "__main__":
    df = run()
    plot(df)

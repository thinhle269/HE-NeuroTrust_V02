"""Attack-robustness study: how each defence holds up under SOTA attacks.

The main experiment (``run_all.py``) uses an uncoordinated ``sign_flip``
attack, under which - at 30% Byzantine - the honest majority absorbs the
damage and every defence merely *matches* vanilla FedAvg.  This study
answers the sharper reviewer question: *"what happens under attacks that
are actually designed to defeat robust aggregators?"*

We sweep the attack taxonomy

    sign_flip  (uncoordinated baseline)
    ipm        Inner Product Manipulation (Xie et al. 2020)
    alie       A Little Is Enough (Baruch et al. 2019)
    min_max    Min-Max agnostic (Shejwalkar & Houmansadr 2021)
    min_sum    Min-Sum agnostic (Shejwalkar & Houmansadr 2021)

over a focused set of defences

    fedavg_attack, fedmedian, krum, bulyan, foolsgold,
    full_system, full_system_neuro

at the nominal 30% Byzantine fraction.  The output table + heat-map show
which defences survive which attacks; the expectation (and the paper's
argument) is that the coordinated attacks break the distance/rank-based
plaintext defences while the trust-driven full_system variants degrade
more gracefully.

Usage
-----
    python run_attack_study.py                       # full study
    python run_attack_study.py --attacks alie ipm    # subset
    python run_attack_study.py --rounds 20 --seeds 42 123
"""
from __future__ import annotations

import matplotlib                                              # noqa: E402
matplotlib.use("Agg")                                          # noqa: E402

import argparse
import copy
import json
import time
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import get_logger, load_config, set_global_seed
from run_all import run_experiment
from run_ablation import set_nested, override_output_dirs

PROJECT_ROOT = Path(__file__).resolve().parent
STUDY_ROOT = PROJECT_ROOT / "results" / "attack_study"

ATTACKS = ["sign_flip", "ipm", "alie", "min_max", "min_sum"]
SCENARIOS = ["fedavg_attack", "fedmedian", "krum", "bulyan", "foolsgold",
             "full_system", "full_system_neuro"]


def run_attack(attack: str, base_cfg, rounds: int, seeds: List[int],
               scenarios: Sequence[str], logger) -> pd.DataFrame:
    run_root = STUDY_ROOT / f"attack_{attack}"
    run_root.mkdir(parents=True, exist_ok=True)
    cfg = copy.deepcopy(base_cfg)
    set_nested(cfg, "experiments.malicious_clients.attack_type", attack)
    cfg.federated.rounds = int(rounds)
    cfg.seeds = [int(s) for s in seeds]
    override_output_dirs(cfg, run_root)
    t0 = time.time()
    try:
        run_experiment(cfg, scenarios_filter=list(scenarios),
                       seeds_override=seeds, force_preprocess=False,
                       logger_name=f"attack[{attack}]")
    except Exception as exc:  # keep the sweep alive
        logger.error("attack %s crashed: %s", attack, exc, exc_info=True)
        return pd.DataFrame()
    logger.info("attack=%-9s done in %.1fs", attack, time.time() - t0)
    per_seed = run_root / "csv" / "scenario_per_seed.csv"
    if not per_seed.exists():
        return pd.DataFrame()
    df = pd.read_csv(per_seed)
    df.insert(0, "attack", attack)
    return df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SOTA attack-robustness study")
    p.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    p.add_argument("--attacks", nargs="+", default=ATTACKS, choices=ATTACKS)
    p.add_argument("--scenarios", nargs="+", default=SCENARIOS)
    p.add_argument("--rounds", type=int, default=30)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2024])
    return p.parse_args()


def main():
    args = parse_args()
    base = load_config(args.config)
    set_global_seed(base.seed)
    STUDY_ROOT.mkdir(parents=True, exist_ok=True)
    logger = get_logger("attack_study", STUDY_ROOT / "logs", level=base.logging.level)
    logger.info("=" * 72)
    logger.info("Attack-robustness study | attacks=%s | scenarios=%s | seeds=%s rounds=%d",
                args.attacks, args.scenarios, args.seeds, args.rounds)

    frames = []
    for atk in args.attacks:
        df = run_attack(atk, base, args.rounds, args.seeds, args.scenarios, logger)
        if not df.empty:
            frames.append(df)
            frames_all = pd.concat(frames, ignore_index=True)
            frames_all.to_csv(STUDY_ROOT / "attack_study_raw.csv", index=False)  # checkpoint

    if not frames:
        logger.warning("no attack results produced")
        return
    raw = pd.concat(frames, ignore_index=True)
    raw.to_csv(STUDY_ROOT / "attack_study_raw.csv", index=False)

    agg = (raw.groupby(["attack", "scenario"])["test_macro_f1"]
           .agg(["mean", "std"]).reset_index())
    agg.to_csv(STUDY_ROOT / "attack_study_summary.csv", index=False)

    pivot = agg.pivot(index="scenario", columns="attack", values="mean")
    pivot = pivot.reindex(index=[s for s in args.scenarios if s in pivot.index],
                          columns=[a for a in args.attacks if a in pivot.columns])
    pivot.to_csv(STUDY_ROOT / "attack_study_pivot.csv")
    logger.info("Macro-F1 pivot (rows=defence, cols=attack):\n%s",
                pivot.round(3).to_string())

    sns.set_theme(style="white", context="paper", font_scale=1.0)
    fig, ax = plt.subplots(figsize=(1.4 * len(pivot.columns) + 3,
                                    0.6 * len(pivot.index) + 2))
    sns.heatmap(pivot.astype(float), annot=True, fmt=".3f", cmap="RdYlGn",
                vmin=0.0, vmax=float(np.nanmax(pivot.values)), ax=ax,
                linewidths=0.5, cbar_kws={"label": "test macro-F1"})
    ax.set_title("Defence robustness across SOTA attacks (30% Byzantine)")
    ax.set_xlabel("attack"); ax.set_ylabel("defence")
    fig.savefig(STUDY_ROOT / "attack_study_heatmap.png", dpi=300, bbox_inches="tight")
    fig.savefig(STUDY_ROOT / "attack_study_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)

    (STUDY_ROOT / "index.json").write_text(json.dumps({
        "attacks": args.attacks, "scenarios": args.scenarios,
        "seeds": args.seeds, "rounds": args.rounds,
    }, indent=2), encoding="utf-8")
    logger.info("Attack study done. Outputs under %s", STUDY_ROOT)


if __name__ == "__main__":
    main()

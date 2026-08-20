"""Parameter-sweep ablation runner.

For each hyperparameter the manuscript needs to justify, this script:

1.  Loads the base config.
2.  For each value in the configured sweep, makes a deep copy of the config,
    sets the parameter, and reruns the requested scenarios under it.  Each
    sub-run writes to ``results/ablation/<param>/value_<v>/`` so artefacts
    never overwrite the main experiment.
3.  Aggregates one summary row per (param_value, scenario, seed) into a
    single sweep CSV.
4.  Renders a publication-quality figure (mean +/- std band per scenario)
    with the *chosen* operating point highlighted by a dashed vertical line.

Ablations implemented out-of-the-box
------------------------------------

* ``trust_threshold``   - tau in [0.10, 0.20, ..., 0.80] for ``full_system``.
                          Optimises the Zero-Trust acceptance threshold.
* ``malicious_fraction``- Byzantine fraction in [0.0, 0.1, ..., 0.5] for the
                          full comparison set (FedAvg / FedMedian / Krum /
                          full_system).  Produces the robustness-vs-attack
                          curve required by Byzantine-FL reviewers.
* ``dirichlet_alpha``   - Non-IID severity in [0.1, 0.3, 0.5, 1.0, 5.0].
                          Shows the method holds under heterogeneous data.
* ``ema_alpha``         - Zero-Trust EMA reactiveness in [0.0, 0.2, ..., 1.0].
                          Bonus ablation to defend the EMA smoothing choice.

Usage
-----

    # one parameter
    python run_ablation.py --param trust_threshold

    # several parameters in one go
    python run_ablation.py --param trust_threshold malicious_fraction

    # custom sweep range
    python run_ablation.py --param trust_threshold --values 0.30 0.40 0.50

    # cheaper sweep (fewer rounds / single seed) for iterative tuning
    python run_ablation.py --param trust_threshold --rounds 15 --seeds 42
"""
from __future__ import annotations

import matplotlib                                              # noqa: E402
matplotlib.use("Agg")                                          # noqa: E402

import argparse
import copy
import json
import shutil
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from src.utils import get_logger, load_config, set_global_seed
from src.evaluation import plot_ablation_curve
from run_all import run_experiment


PROJECT_ROOT = Path(__file__).resolve().parent


ABLATIONS: Dict[str, Dict] = {
    "trust_threshold": {
        "path": "zero_trust.trust_threshold",
        "values": [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80],
        "scenarios": ["full_system"],
        "chosen": 0.40,
        "description": "Zero-Trust acceptance threshold (tau)",
    },
    "malicious_fraction": {
        "path": "experiments.malicious_clients.fraction",
        "values": [0.00, 0.10, 0.20, 0.30, 0.40, 0.50],
        "scenarios": ["fedavg_attack", "fedmedian", "krum", "full_system"],
        "chosen": 0.30,
        "description": "Fraction of Byzantine clients",
    },
    "dirichlet_alpha": {
        "path": "federated.dirichlet_alpha",
        "values": [0.10, 0.30, 0.50, 1.00, 5.00],
        "scenarios": ["fedavg_attack", "fedmedian", "full_system"],
        "chosen": 0.50,
        "description": "Dirichlet alpha (data heterogeneity; smaller = more non-IID)",
    },
    "ema_alpha": {
        "path": "zero_trust.ema_alpha",
        "values": [0.20, 0.40, 0.60, 0.80, 1.00],
        "scenarios": ["full_system"],
        "chosen": 0.60,
        "description": "EMA weight on the newest trust sample",
    },
}


def set_nested(cfg, path: str, value):
    """Set ``cfg.a.b.c = value`` via the dotted ``path``."""
    parts = path.split(".")
    obj = cfg
    for p in parts[:-1]:
        if hasattr(obj, p):
            obj = getattr(obj, p)
        else:
            obj = obj[p]
    last = parts[-1]
    if hasattr(obj, "__setitem__"):
        obj[last] = value
    else:
        setattr(obj, last, value)


def override_output_dirs(cfg, root: Path) -> None:
    """Redirect every results subdir to a private folder under ``root``."""
    cfg.paths.results_dir = str(root)
    cfg.paths.figures_dir = str(root / "figures")
    cfg.paths.csv_dir = str(root / "csv")
    cfg.paths.models_dir = str(root / "models")
    cfg.paths.logs_dir = str(root / "logs")


def reduce_for_ablation(cfg, rounds: int, seeds: List[int]) -> None:
    """Make a sweep tractable by trimming rounds and using fewer seeds.

    Ablation figures show *trends* across the parameter axis, so a slightly
    shorter run is fine here - we keep full settings for the main results
    table in ``run_all.py``.
    """
    cfg.federated.rounds = int(rounds)
    cfg.seeds = [int(s) for s in seeds]


def run_single_sweep(param: str, base_cfg, rounds: int, seeds: List[int],
                     override_values: Sequence = None,
                     logger=None) -> pd.DataFrame:
    """Sweep one parameter; return a tidy DataFrame.

    Columns: ``param_name, param_value, scenario, seed,
              test_accuracy, test_macro_f1, test_weighted_f1, test_loss``.
    """
    spec = ABLATIONS[param]
    values = list(override_values) if override_values else list(spec["values"])
    scenarios = list(spec["scenarios"])
    ablation_root = PROJECT_ROOT / "results" / "ablation" / param
    ablation_root.mkdir(parents=True, exist_ok=True)
    logger = logger or get_logger(f"ablation[{param}]", ablation_root / "logs",
                                  level=base_cfg.logging.level)
    logger.info("=" * 72)
    logger.info("Ablation: %s | %s", param, spec["description"])
    logger.info("Values: %s", values)
    logger.info("Scenarios: %s", scenarios)
    logger.info("Seeds: %s | Rounds: %d", seeds, rounds)

    rows = []
    sweep_csv = ablation_root / "sweep_raw.csv"
    completed_values = set()
    if sweep_csv.exists():
        try:
            prev = pd.read_csv(sweep_csv)
            rows.extend(prev.to_dict("records"))
            completed_values = set(prev["param_value"].astype(float).unique())
            logger.info("Resuming - %d values already in %s: %s",
                        len(completed_values), sweep_csv, sorted(completed_values))
        except Exception as exc:
            logger.warning("Failed to load existing sweep CSV (%s): %s -- "
                           "starting fresh.", sweep_csv, exc)
            rows = []

    failed_values: List[float] = []
    for v in values:
        v_float = float(v)
        if v_float in completed_values:
            logger.info("  value=%-6g already done -> skipping", v)
            continue
        run_root = ablation_root / f"value_{v:g}"
        run_root.mkdir(parents=True, exist_ok=True)
        cfg = copy.deepcopy(base_cfg)
        set_nested(cfg, spec["path"], v)
        reduce_for_ablation(cfg, rounds=rounds, seeds=seeds)
        override_output_dirs(cfg, run_root)
        t0 = time.time()
        try:
            result = run_experiment(cfg, scenarios_filter=scenarios,
                                    seeds_override=seeds,
                                    force_preprocess=False,
                                    logger_name=f"ablation[{param}={v:g}]")
            elapsed = time.time() - t0
            logger.info("  value=%-6g done in %.1fs", v, elapsed)
        except Exception as exc:
            elapsed = time.time() - t0
            logger.error("  value=%-6g FAILED after %.1fs: %s -- continuing "
                         "with the rest of the sweep.", v, elapsed, exc,
                         exc_info=True)
            failed_values.append(v_float)
            continue

        per_seed_path = run_root / "csv" / "scenario_per_seed.csv"
        if per_seed_path.exists():
            ps = pd.read_csv(per_seed_path)
            for _, r in ps.iterrows():
                rows.append({
                    "param_name": param,
                    "param_value": v_float,
                    "scenario": r["scenario"],
                    "seed": int(r["seed"]),
                    "test_accuracy": float(r["test_accuracy"]),
                    "test_macro_f1": float(r["test_macro_f1"]),
                    "test_weighted_f1": float(r["test_weighted_f1"]),
                    "test_loss": float(r["test_loss"]),
                })
            pd.DataFrame(rows).to_csv(sweep_csv, index=False)

    sweep = pd.DataFrame(rows)
    sweep.to_csv(sweep_csv, index=False)
    logger.info("Raw sweep -> %s (%d rows; %d points failed: %s)",
                sweep_csv, len(sweep), len(failed_values), failed_values)
    return sweep


def aggregate_and_plot(param: str, sweep: pd.DataFrame, logger=None) -> pd.DataFrame:
    """Aggregate per-(value, scenario) over seeds and render the curve figure."""
    spec = ABLATIONS[param]
    ablation_root = PROJECT_ROOT / "results" / "ablation" / param
    metric_cols = ["test_accuracy", "test_macro_f1", "test_weighted_f1", "test_loss"]
    if sweep.empty:
        if logger:
            logger.warning("[ablation:%s] sweep DataFrame is empty - skipping plot", param)
        return sweep

    grouped = (sweep
               .groupby(["param_value", "scenario"])[metric_cols]
               .agg(["mean", "std", "count"])
               .reset_index())
    flat_cols = []
    for col in grouped.columns:
        if isinstance(col, tuple):
            head, agg = col
            flat_cols.append(head if not agg else f"{head}_{agg}")
        else:
            flat_cols.append(col)
    grouped.columns = flat_cols
    out_csv = ablation_root / "sweep_summary.csv"
    grouped.to_csv(out_csv, index=False)
    if logger:
        logger.info("Sweep summary -> %s", out_csv)

    for metric in ("test_accuracy", "test_macro_f1"):
        plot_ablation_curve(
            grouped, ablation_root / f"sweep_{metric}",
            param_name=param, metric=metric,
            title=f"Ablation on {spec['description']}: {metric}",
            chosen_value=spec.get("chosen"),
        )
        if logger:
            logger.info("Figure -> %s.{png,pdf}", ablation_root / f"sweep_{metric}")
    return grouped


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FL-IDS hyperparameter ablation runner")
    p.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    p.add_argument("--param", nargs="+", default=list(ABLATIONS.keys()),
                   choices=list(ABLATIONS.keys()),
                   help="Which ablation(s) to run. Default = all four.")
    p.add_argument("--values", nargs="+", type=float, default=None,
                   help="Override the configured sweep values (only when "
                        "--param specifies exactly one).")
    p.add_argument("--rounds", type=int, default=15,
                   help="FL rounds per ablation point.  Default 15 - this is "
                        "the setting used for the threshold-sensitivity "
                        "diagnostic reported in the paper (the main run uses "
                        "30).  Ablation figures are about *trends*, not "
                        "absolute peak accuracy.")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 123],
                   help="Seeds per ablation point.  Default 2 - keep cheap.")
    return p.parse_args()


def main():
    args = parse_args()
    if args.values is not None and len(args.param) > 1:
        raise SystemExit("--values can only be combined with a single --param")

    base = load_config(args.config)
    set_global_seed(base.seed)

    root_logger = get_logger("ablation_main",
                             PROJECT_ROOT / "results" / "ablation" / "logs",
                             level=base.logging.level)

    summary_index = []
    for param in args.param:
        try:
            sweep = run_single_sweep(param, base,
                                     rounds=args.rounds, seeds=args.seeds,
                                     override_values=args.values,
                                     logger=root_logger)
            agg = aggregate_and_plot(param, sweep, logger=root_logger)
            n_points = int(agg["param_value"].nunique()) if not agg.empty else 0
            status = "ok"
            error = None
        except Exception as exc:
            root_logger.error("Ablation [%s] crashed: %s -- skipping but "
                              "continuing with the next ablation.",
                              param, exc, exc_info=True)
            n_points = 0
            status = "failed"
            error = str(exc)
        summary_index.append({
            "param": param,
            "description": ABLATIONS[param]["description"],
            "values": ABLATIONS[param]["values"]
                       if args.values is None else list(args.values),
            "scenarios": ABLATIONS[param]["scenarios"],
            "chosen": ABLATIONS[param].get("chosen"),
            "n_seeds": len(args.seeds),
            "n_rounds": args.rounds,
            "n_points": n_points,
            "status": status,
            "error": error,
        })
        idx_path = PROJECT_ROOT / "results" / "ablation" / "index.json"
        idx_path.parent.mkdir(parents=True, exist_ok=True)
        idx_path.write_text(json.dumps(summary_index, indent=2, default=str),
                            encoding="utf-8")

    idx_path = PROJECT_ROOT / "results" / "ablation" / "index.json"
    idx_path.write_text(json.dumps(summary_index, indent=2, default=str),
                        encoding="utf-8")
    root_logger.info("All ablations done. Index -> %s", idx_path)


if __name__ == "__main__":
    main()

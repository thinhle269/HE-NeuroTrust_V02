"""Paired significance tests between the proposed system and each baseline.

Reads ``results/csv/scenario_per_seed.csv`` (one row per scenario x seed)
and, for the reference scenario (default ``full_system``), runs a paired
comparison against every other scenario across the shared seeds:

* paired Wilcoxon signed-rank test (non-parametric, robust to small n),
* paired t-test (parametric companion),
* mean difference and Cohen's d effect size.

With 5 seeds the tests are necessarily low-powered; we report them for
completeness and lean on effect size + consistent sign of the difference,
which is the honest position for a small-n FL study.

Usage
-----
    python analyze_significance.py                     # ref = full_system
    python analyze_significance.py --ref full_system_neuro --metric test_macro_f1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent


def cohens_d(diff: np.ndarray) -> float:
    if diff.std(ddof=1) == 0:
        return 0.0
    return float(diff.mean() / diff.std(ddof=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(PROJECT_ROOT / "results" / "csv" / "scenario_per_seed.csv"))
    ap.add_argument("--ref", default="full_system")
    ap.add_argument("--metric", default="test_macro_f1",
                    choices=["test_macro_f1", "test_accuracy", "test_weighted_f1"])
    ap.add_argument("--out", default=str(PROJECT_ROOT / "results" / "csv" / "significance.csv"))
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.ref not in df.scenario.unique():
        raise SystemExit(f"reference scenario '{args.ref}' not in {sorted(df.scenario.unique())}")

    ref = df[df.scenario == args.ref].set_index("seed")[args.metric]
    rows = []
    for scen in sorted(df.scenario.unique()):
        if scen == args.ref:
            continue
        other = df[df.scenario == scen].set_index("seed")[args.metric]
        seeds = sorted(set(ref.index) & set(other.index))
        if len(seeds) < 2:
            continue
        a = ref.loc[seeds].to_numpy()
        b = other.loc[seeds].to_numpy()
        diff = a - b
        try:
            w_stat, w_p = stats.wilcoxon(a, b)
        except ValueError:
            w_stat, w_p = np.nan, np.nan
        t_stat, t_p = stats.ttest_rel(a, b)
        rows.append({
            "reference": args.ref,
            "vs_scenario": scen,
            "metric": args.metric,
            "n_seeds": len(seeds),
            "ref_mean": round(float(a.mean()), 4),
            "other_mean": round(float(b.mean()), 4),
            "mean_diff": round(float(diff.mean()), 4),
            "wins": int((diff > 0).sum()),
            "wilcoxon_p": round(float(w_p), 4) if np.isfinite(w_p) else np.nan,
            "ttest_p": round(float(t_p), 4) if np.isfinite(t_p) else np.nan,
            "cohens_d": round(cohens_d(diff), 3),
        })
    out = pd.DataFrame(rows).sort_values("mean_diff", ascending=False)
    out.to_csv(args.out, index=False)
    print(f"Reference = {args.ref} | metric = {args.metric} | seeds = {len(set(ref.index))}")
    print(out.to_string(index=False))
    print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()

"""Generate Section 4 (Experiments & Results) figures + table CSVs from the
final Adam+50k results.  Publication style, unified palette.

Figures (new_paper/figures/):
  fig3_convergence.{png,pdf}    - validation macro-F1 vs round, key scenarios
  fig4_headline_bars.{png,pdf}  - final test macro-F1 / weighted-F1 per scenario
  fig5_attack_heatmap.{png,pdf} - defence x SOTA-attack macro-F1
  fig6_privacy.{png,pdf}        - gradient-inversion (DLG) leakage
  fig7_multidataset.{png,pdf}   - best-defence vs no-defence across 4 datasets

Tables (new_paper/tables/):
  t1_headline.csv, t3_attack.csv, t5_multidataset.csv  - tidy CSVs used by the
  docx builder / for the record.

Run:  python new_paper/make_assets/make_section4_assets.py
"""
from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "new_paper" / "figures"
TAB = ROOT / "new_paper" / "tables"
FIG.mkdir(parents=True, exist_ok=True); TAB.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)

PROP = {"full_system_neuro", "full_system", "fedavg_he_fuzzy", "fedavg_he"}
LABEL = {
    "centralized": "Centralized", "fedavg": "FedAvg (clean)",
    "fedavg_attack": "FedAvg + attack", "fedavg_he": "FedAvg + HE",
    "fedprox": "FedProx", "foolsgold": "FoolsGold", "krum": "Krum",
    "trimmed_mean": "Trimmed-Mean", "fedmedian": "Median", "bulyan": "Bulyan",
    "fedavg_he_fuzzy": "HE + Fuzzy (Mamdani)", "full_system": "HE-NeuroTrust (Mamdani)",
    "full_system_neuro": "HE-NeuroTrust (neuro-fuzzy)",
}
def _save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_convergence():
    pr = pd.read_csv(ROOT / "results/csv/per_round_metrics.csv")
    keep = ["fedavg", "fedavg_attack", "krum", "bulyan", "full_system", "full_system_neuro"]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    pal = sns.color_palette("tab10", len(keep))
    for i, s in enumerate(keep):
        sub = pr[pr.scenario == s]
        if sub.empty:
            continue
        g = sub.groupby("round_idx")["val_macro_f1"].agg(["mean", "std"]).reset_index()
        x = g.round_idx + 1
        style = "-" if s in PROP else "--"
        lw = 2.4 if s in PROP else 1.5
        ax.plot(x, g["mean"], style, color=pal[i], lw=lw, marker="o", ms=3, label=LABEL.get(s, s))
        ax.fill_between(x, g["mean"] - g["std"], g["mean"] + g["std"], color=pal[i], alpha=0.12)
    ax.set_xlabel("federated round"); ax.set_ylabel("validation macro-F1")
    ax.set_title("Convergence under 30% sign-flip (mean ± std, 5 seeds)", fontsize=10, weight="bold")
    ax.legend(fontsize=8, ncol=2, loc="lower right")
    _save(fig, "fig3_convergence")
    print("fig3 ok")


def fig_headline():
    d = pd.read_csv(ROOT / "results/csv/scenario_summary.csv")
    order = d.sort_values("test_macro_f1_mean", ascending=False).scenario.tolist()
    order = [s for s in order if s != "centralized"]
    d = d.set_index("scenario").loc[order].reset_index()
    x = np.arange(len(d)); w = 0.4
    fig, ax = plt.subplots(figsize=(10, 4.6))
    c1 = ["#1b5e20" if s in PROP else "#78909c" for s in d.scenario]
    c2 = ["#43a047" if s in PROP else "#b0bec5" for s in d.scenario]
    ax.bar(x - w/2, d.test_macro_f1_mean, w, yerr=d.test_macro_f1_std, capsize=3,
           color=c1, edgecolor="black", linewidth=0.4, label="macro-F1")
    ax.bar(x + w/2, d.test_weighted_f1_mean, w, yerr=d.test_weighted_f1_std, capsize=3,
           color=c2, edgecolor="black", linewidth=0.4, label="weighted-F1")
    ax.set_xticks(x); ax.set_xticklabels([LABEL.get(s, s) for s in d.scenario], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("test F1"); ax.set_ylim(0, 0.85)
    ax.set_title("CIC-IoT-2023 final test F1 (5 seeds; green = HE-NeuroTrust variants)",
                 fontsize=10, weight="bold")
    ax.legend(fontsize=9)
    _save(fig, "fig4_headline_bars")
    print("fig4 ok")


def fig_attack():
    p = ROOT / "results/attack_study/attack_study_pivot.csv"
    piv = pd.read_csv(p, index_col=0)
    row_order = [r for r in ["fedavg_attack", "fedmedian", "krum", "bulyan", "full_system", "full_system_neuro"] if r in piv.index]
    piv = piv.loc[row_order]
    piv.index = [LABEL.get(r, r) for r in piv.index]        # paper-ready row labels
    _atk = {"sign_flip": "Sign-flip", "ipm": "IPM", "alie": "ALIE", "min_max": "Min-Max", "min_sum": "Min-Sum"}
    piv.columns = [_atk.get(c, c) for c in piv.columns]     # paper-ready column labels
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    sns.heatmap(piv.astype(float), annot=True, fmt=".3f", cmap="RdYlGn",
                vmin=float(piv.values.min()), vmax=float(piv.values.max()),
                linewidths=0.5, cbar_kws={"label": "test macro-F1"}, ax=ax)
    ax.set_title("Robustness to coordinated attacks", fontsize=10, weight="bold")
    ax.set_xlabel("attack"); ax.set_ylabel("defence")
    _save(fig, "fig5_attack_heatmap")
    print("fig5 ok")


def fig_privacy():
    """DLG leakage: per-trial distribution, not just a mean bar.

    Shows every individual reconstruction trial (strip) on top of the mean with a
    95% bootstrap CI, with the y-axis fixed to the valid cosine range [-1, 1] so
    the uncertainty cannot appear to leave the domain (review item P1-07).
    """
    order = ["plaintext_single", "plaintext_batch", "he_aggregate"]
    labels = ["plaintext\n(single update)", "plaintext\n(batch-averaged)",
              "decrypted aggregate\n(ours)"]
    cols = ["#d62728", "#ff7f0e", "#2ca02c"]
    raw_p = ROOT / "results/privacy_eval/privacy_raw.csv"
    summ = pd.read_csv(ROOT / "results/privacy_eval/privacy_summary.csv").set_index("regime")
    raw = pd.read_csv(raw_p) if raw_p.exists() else None

    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    n_trials = None
    for i, reg in enumerate(order):
        mean = float(summ.loc[reg, "cosine_mean"])
        ax.bar(i, mean, 0.62, color=cols[i], edgecolor="black", linewidth=0.6,
               alpha=0.85, zorder=2)
        if raw is not None:
            vals = raw.loc[raw.regime == reg, "cosine"].dropna().to_numpy(dtype=float)
            if {"cosine_ci_lo", "cosine_ci_hi", "n_valid"}.issubset(summ.columns):
                lo = float(summ.loc[reg, "cosine_ci_lo"])
                hi = float(summ.loc[reg, "cosine_ci_hi"])
                n_trials = int(summ.loc[reg, "n_valid"])
            else:                                    # legacy summary without CI
                n_trials = len(vals)
                boot = np.array([rng.choice(vals, len(vals), replace=True).mean()
                                 for _ in range(2000)])
                lo, hi = np.percentile(boot, [2.5, 97.5])
            ax.errorbar(i, mean, yerr=[[mean - lo], [hi - mean]], fmt="none",
                        ecolor="black", elinewidth=1.3, capsize=5, zorder=4)
            ax.scatter(i + rng.uniform(-0.16, 0.16, len(vals)), vals, s=13,
                       facecolor="white", edgecolor="#333333", linewidth=0.6,
                       alpha=0.85, zorder=5)
        ci_txt = f"\n[{lo:+.2f}, {hi:+.2f}]  n={n_trials}" if raw is not None else ""
        ax.text(i, -0.90, f"mean {mean:+.3f}{ci_txt}", ha="center", va="center",
                fontsize=9, weight="bold", zorder=6,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor="#bbbbbb", alpha=0.95))

    ax.axhline(0, color="gray", lw=0.8, zorder=1)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(-1.0, 1.0)                      # valid cosine domain
    ax.set_ylabel("reconstruction cosine similarity\n(higher = more leakage)")
    sub = ("points = individual DLG trials (n per group shown); bar = mean; "
           "error bar = 95% bootstrap CI" if n_trials else "bar = mean over DLG trials")
    ax.set_title("Gradient-inversion (DLG) leakage by server access level\n" + sub,
                 fontsize=9.5, weight="bold")
    ax.grid(True, axis="y", alpha=0.3, zorder=0)
    fig.tight_layout()
    _save(fig, "fig6_privacy")
    print("fig6 ok")


def fig_multidataset():
    rows = []
    specs = [("CIC-IoT-2023", "results"), ("Edge-IIoT", "results_edgeiiot"),
             ("keystroke", "results_keystroke"), ("HMOG", "results_hmog")]
    for name, d in specs:
        f = ROOT / d / "csv" / "scenario_summary.csv"
        if not f.exists():
            continue
        s = pd.read_csv(f).set_index("scenario")
        ours = [x for x in ("full_system", "full_system_neuro") if x in s.index]
        best = s.loc[ours, "test_macro_f1_mean"].idxmax()
        rows.append({"dataset": name, "best_ours": s.loc[best, "test_macro_f1_mean"],
                     "no_defence": s.loc["fedavg_attack", "test_macro_f1_mean"] if "fedavg_attack" in s.index else np.nan})
    d = pd.DataFrame(rows)
    x = np.arange(len(d)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar(x - w/2, d.best_ours, w, color="#1b5e20", edgecolor="black", label="HE-NeuroTrust (best)")
    ax.bar(x + w/2, d.no_defence, w, color="#c62828", edgecolor="black", label="no defence (FedAvg+attack)")
    ax.set_xticks(x); ax.set_xticklabels(d.dataset, fontsize=9)
    ax.set_ylabel("test macro-F1"); ax.set_ylim(0, 0.95)
    for i, (a, b) in enumerate(zip(d.best_ours, d.no_defence)):
        ax.text(i - w/2, a + 0.01, f"{a:.2f}", ha="center", fontsize=8)
        ax.text(i + w/2, b + 0.01, f"{b:.2f}", ha="center", fontsize=8)
    ax.set_title("Multi-domain generalisation (best defence vs no defence, 30% attack)",
                 fontsize=10, weight="bold")
    ax.legend(fontsize=9)
    _save(fig, "fig7_multidataset")
    d.to_csv(TAB / "t5_multidataset.csv", index=False)
    print("fig7 ok")


if __name__ == "__main__":
    fig_convergence(); fig_headline(); fig_attack(); fig_privacy(); fig_multidataset()
    print("Section 4 assets ->", FIG)

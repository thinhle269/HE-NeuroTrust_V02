"""Section 5.4 sensitivity figure: Zero-Trust threshold tau sweep.

fig8_tau_sensitivity.{png,pdf} - test macro-F1 / weighted-F1 vs tau from the
real ablation sweep (results/ablation/trust_threshold/sweep_summary.csv),
showing the broad stable plateau for tau in [0.1, 0.4] and the collapse for
tau >= 0.5 when the Zero-Trust gate over-rejects and the participation floor
dominates.  The operating point tau = 0.40 is marked.

Run:  python new_paper/make_assets/make_section5_tau.py
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "new_paper" / "figures"; FIG.mkdir(parents=True, exist_ok=True)
TAB = ROOT / "new_paper" / "tables"; TAB.mkdir(parents=True, exist_ok=True)
SRC = ROOT / "results" / "ablation" / "trust_threshold" / "sweep_summary.csv"


def main():
    d = pd.read_csv(SRC).sort_values("param_value").reset_index(drop=True)
    tau = d.param_value.to_numpy()
    mf1, mf1s = d.test_macro_f1_mean.to_numpy(), d.test_macro_f1_std.to_numpy()
    wf1, wf1s = d.test_weighted_f1_mean.to_numpy(), d.test_weighted_f1_std.to_numpy()

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.axvspan(0.10, 0.80, color="#2ca02c", alpha=0.07, zorder=0)
    ax.text(0.60, 0.09, "stable across the full range\n(participation floor prevents collapse)",
            ha="center", fontsize=8.5, color="#1b5e20", style="italic")

    ax.errorbar(tau, mf1, yerr=mf1s, marker="o", ms=5, lw=2.0, capsize=3,
                color="#1b5e20", label="test macro-F1")
    ax.errorbar(tau, wf1, yerr=wf1s, marker="s", ms=5, lw=2.0, capsize=3,
                color="#43a047", ls="--", label="test weighted-F1")

    op = np.argmin(np.abs(tau - 0.40))
    ax.axvline(0.40, color="#1f3864", lw=1.4, ls=":")
    ax.scatter([0.40], [mf1[op]], s=120, facecolor="none", edgecolor="#1f3864",
               linewidth=2.0, zorder=6)
    ax.annotate("operating point\n$\\tau = 0.40$", xy=(0.40, mf1[op]),
                xytext=(0.52, 0.42), fontsize=9, color="#1f3864", weight="bold",
                arrowprops=dict(arrowstyle="->", color="#1f3864", lw=1.2))

    ax.set_xlabel("Zero-Trust acceptance threshold  $\\tau$")
    ax.set_ylabel("test F1")
    ax.set_ylim(0.0, 0.85)
    ax.set_xlim(0.05, 0.85)
    ax.set_title("Sensitivity to the Zero-Trust threshold $\\tau$ "
                 "(CIC-IoT-2023, full framework)", fontsize=10, weight="bold")
    ax.legend(fontsize=9, loc="lower left", framealpha=0.95)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig8_tau_sensitivity.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / "fig8_tau_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)

    out = d[["param_value", "test_macro_f1_mean", "test_macro_f1_std",
             "test_weighted_f1_mean", "test_weighted_f1_std",
             "test_accuracy_mean", "test_accuracy_std"]].copy()
    out.columns = ["tau", "macro_f1", "macro_f1_std", "weighted_f1",
                   "weighted_f1_std", "accuracy", "accuracy_std"]
    out.to_csv(TAB / "t6_tau_sensitivity.csv", index=False)
    print("fig8_tau_sensitivity ok ->", FIG)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()

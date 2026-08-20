"""Section 5 HE-overhead analysis from real per-round timing.

Reads results/csv/per_round_metrics.csv (all seeds) and produces:
  fig_overhead.{png,pdf}        - per-round wall-clock breakdown per scenario
  new_paper/tables/t_overhead.csv - tidy overhead table (mean +/- std)

The message: HE inflates the round ~1.5-1.7x (encryption-dominated), the
neuro-fuzzy trust engine is essentially free (<1% of the round), and the
64x Paillier ciphertext expansion is what forces the compact-model design.

Run:  python new_paper/make_assets/make_section5_overhead.py
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
N_PARAMS = 4712          # tabular MLP [64,32]
BYTES_FP32 = N_PARAMS * 4
N_CLIENTS = 10

COMPONENTS = [
    ("time_local_train_sec", "Local training", "#8fb4d9"),
    ("time_encrypt_sec",     "Paillier encrypt", "#e08214"),
    ("time_aggregate_sec",   "HE aggregate", "#fdb863"),
    ("time_decrypt_sec",     "Decrypt", "#b2182b"),
    ("time_fuzzy_sec",       "Trust engine", "#2ca02c"),
]
SCEN_ORDER = [
    ("fedavg",            "FedAvg\n(plaintext)"),
    ("krum",              "Krum\n(plaintext robust)"),
    ("fedavg_he",         "FedAvg+HE\n(privacy only)"),
    ("full_system",       "HE-NeuroTrust\n(Mamdani)"),
    ("full_system_neuro", "HE-NeuroTrust\n(neuro-fuzzy)"),
]


def main():
    d = pd.read_csv(ROOT / "results/csv/per_round_metrics.csv")
    rows = []
    for scen, _ in SCEN_ORDER:
        s = d[d.scenario == scen]
        if s.empty:
            continue
        rec = {"scenario": scen}
        for col, _, _ in COMPONENTS:
            rec[col] = float(s[col].mean())
        rec["time_total_sec"] = float(s["time_total_sec"].mean())
        rec["total_std"] = float(s.groupby("seed")["time_total_sec"].mean().std())
        rec["he_ciphertext_MB"] = float(s["he_ciphertext_bytes"].mean()) / 1e6
        rec["mal_rej"] = float(s["n_malicious_rejected"].mean())
        rec["mal_tot"] = float(s["n_malicious_total"].mean())
        rows.append(rec)
    o = pd.DataFrame(rows).set_index("scenario")

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    labels = [lab for scen, lab in SCEN_ORDER if scen in o.index]
    scens = [scen for scen, _ in SCEN_ORDER if scen in o.index]
    x = np.arange(len(scens))
    bottom = np.zeros(len(scens))
    for col, name, color in COMPONENTS:
        vals = o.loc[scens, col].to_numpy()
        ax.bar(x, vals, bottom=bottom, color=color, edgecolor="black", linewidth=0.4, label=name)
        bottom += vals
    for i, scen in enumerate(scens):
        ax.text(i, bottom[i] + 0.35, f"{o.loc[scen,'time_total_sec']:.1f}s",
                ha="center", fontsize=9, weight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("wall-clock time per federated round (s)")
    ax.set_ylim(0, bottom.max() * 1.22)
    ax.set_title("Per-round computational overhead breakdown (CIC-IoT-2023, 5 seeds).\n"
                 "HE encryption dominates; the neuro-fuzzy trust engine is negligible.",
                 fontsize=10, weight="bold", pad=14)
    ax.legend(fontsize=8.5, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.13))
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig_overhead.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / "fig_overhead.pdf", bbox_inches="tight")
    plt.close(fig)

    tbl = o.copy()
    tbl["expansion"] = (tbl["he_ciphertext_MB"] * 1e6 / N_CLIENTS) / BYTES_FP32
    keep = tbl[["time_local_train_sec", "time_encrypt_sec", "time_aggregate_sec",
                "time_decrypt_sec", "time_fuzzy_sec", "time_total_sec",
                "he_ciphertext_MB", "expansion", "mal_rej", "mal_tot"]].round(3)
    keep.to_csv(TAB / "t_overhead.csv")
    print("fig_overhead + t_overhead ok")
    print(keep.to_string())
    if "fedavg" in o.index and "full_system_neuro" in o.index:
        base = o.loc["fedavg", "time_total_sec"]
        full = o.loc["full_system_neuro", "time_total_sec"]
        enc = o.loc["full_system_neuro", "time_encrypt_sec"]
        fuz = o.loc["full_system_neuro", "time_fuzzy_sec"]
        print(f"\nHE slowdown factor    = {full/base:.2f}x ({base:.1f}s -> {full:.1f}s)")
        print(f"encrypt share of round= {enc/full*100:.1f}%")
        print(f"trust-engine share    = {fuz/full*100:.2f}%  ({fuz:.3f}s)")
        print(f"ciphertext / round    = {o.loc['full_system_neuro','he_ciphertext_MB']:.2f} MB "
              f"({o.loc['full_system_neuro','he_ciphertext_MB']/N_CLIENTS*1000:.0f} KB/client)")
        print(f"expansion vs fp32     = {keep.loc['full_system_neuro','expansion']:.0f}x")


if __name__ == "__main__":
    main()

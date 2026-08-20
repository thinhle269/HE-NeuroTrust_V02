"""Generate Section 3 (Proposed Methodology) figures for HE-NeuroTrust.

Outputs (new_paper/figures/):
  fig1_architecture.{png,pdf}      - one-round data flow of the HE-NeuroTrust
                                     framework (client training -> plaintext
                                     attestation -> Paillier encryption ->
                                     neuro-fuzzy trust -> Zero-Trust gate ->
                                     homomorphic aggregation -> decrypt/apply).
  fig2_neurofuzzy.{png,pdf}        - the differentiable ANFIS: learned Gaussian
                                     membership functions per input + the 2-D
                                     trust decision surface.

Run:  python new_paper/make_assets/make_section3_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
FIG = ROOT / "new_paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def _box(ax, xy, w, h, text, fc, ec="#333333", fs=9, tc="black"):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.03",
                                linewidth=1.2, facecolor=fc, edgecolor=ec))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, weight="bold", wrap=True)


def _arrow(ax, p0, p1, color="#444444", style="-|>", lw=1.4, ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=14,
                                 linewidth=lw, color=color, linestyle=ls,
                                 shrinkA=2, shrinkB=2))


def fig_architecture():
    """One federated round, matching the protocol as finally specified:

    * the client computes the directional attestation against a PUBLIC reference,
    * the server holds only the Paillier public key and aggregates ciphertexts by
      product/exponentiation, and
    * a separate decryption authority holds the secret key and opens ONLY the
      aggregate.
    Boxes are sized to their text and the cross-boundary arrows are routed so no
    label sits on a line.
    """
    fig, ax = plt.subplots(figsize=(12, 6.6))
    ax.set_xlim(-0.7, 13); ax.set_ylim(0, 8.6); ax.axis("off")

    CLIENT = "#dbeafe"; CRYPTO = "#fde68a"; TRUST = "#dcfce7"; SERVER = "#fecaca"
    KEYS = "#ede9fe"
    CX, CW = 0.35, 4.05          # client column
    SX, SW = 7.35, 5.10          # server column
    BOUND = 5.95                 # network boundary

    ax.text(CX + CW / 2, 8.05, "CLIENT  i  (IoT device)", ha="center",
            fontsize=11.5, weight="bold")
    _box(ax, (CX, 6.55), CW, 0.85, "1. Local training\n(compact MLP / CNN-LSTM)", CLIENT, fs=9)
    _box(ax, (CX, 5.25), CW, 0.85, "2. Update  Δw_i = w_i − w_g\n(gradient clip + NaN guard)", CLIENT, fs=9)
    _box(ax, (CX, 3.95), CW, 0.85, "3a. Encrypt with public key\nE(Δw_i)  =  PaillierEnc(pk, ·)", CRYPTO, fs=9)
    _box(ax, (CX, 2.60), CW, 0.90, "3b. Attestation  a_i\ncos to public reference, Δloss, vol", TRUST, fs=8.6)

    _arrow(ax, (CX + CW / 2, 6.55), (CX + CW / 2, 6.10))
    _arrow(ax, (CX + CW * 0.35, 5.25), (CX + CW * 0.35, 4.80))
    _arrow(ax, (CX + CW * 0.65, 5.25), (CX + CW * 0.65, 3.50))

    ax.text(SX + SW / 2, 8.05, "SERVER  (honest-but-curious;  public key only)",
            ha="center", fontsize=11.5, weight="bold")
    _box(ax, (SX, 6.55), SW, 0.85, "4. Neuro-fuzzy trust engine\n(differentiable ANFIS)  →  t_i", TRUST, fs=9)
    _box(ax, (SX, 5.25), SW, 0.85, "5. Zero-Trust policy\nEMA + reject-streak + floor  →  α̂_i", TRUST, fs=9)
    _box(ax, (SX, 3.85), SW, 0.95, "6. Homomorphic aggregation\nC = Π  E(Δw_i)^α̂_i   (ciphertexts only)", CRYPTO, fs=9)
    _box(ax, (SX, 1.15), SW, 0.85, "7. Apply update\nw_g ← w_g + Δw_g  (+ rollback guard)", SERVER, fs=9)

    _arrow(ax, (SX + SW / 2, 6.55), (SX + SW / 2, 6.10))
    _arrow(ax, (SX + SW / 2, 5.25), (SX + SW / 2, 4.80))

    _box(ax, (SX, 2.35), SW, 0.95, "Decryption authority  (holds sk)\nopens ONLY the aggregate  →  Δw_g",
         KEYS, fs=9)
    _arrow(ax, (SX + SW * 0.32, 3.85), (SX + SW * 0.32, 3.30), color="#6d28d9")
    ax.text(SX + SW * 0.32 - 0.15, 3.58, "C", fontsize=8.5, color="#6d28d9",
            ha="right", style="italic")
    _arrow(ax, (SX + SW * 0.68, 2.35), (SX + SW * 0.68, 2.00), color="#6d28d9")
    ax.text(SX + SW * 0.68 + 0.15, 2.17, "Δw_g", fontsize=8.5, color="#6d28d9",
            ha="left", style="italic")

    _arrow(ax, (CX + CW, 4.37), (SX, 4.37), color="#b45309", lw=1.6)
    ax.text((CX + CW + SX) / 2, 4.55, "ciphertext  E(Δw_i)", fontsize=8.5,
            color="#b45309", ha="center", style="italic")
    ax.annotate("", xy=(SX, 6.90), xytext=(CX + CW, 3.05),
                arrowprops=dict(arrowstyle="-|>", color="#16a34a", lw=1.6,
                                connectionstyle="angle3,angleA=0,angleB=75",
                                mutation_scale=14))
    ax.text((CX + CW + SX) / 2 + 0.15, 3.25, "plaintext attestation  a_i",
            fontsize=8.5, color="#16a34a", ha="center", style="italic")
    RET_Y, RET_X = 0.72, -0.40
    ret = dict(color="#374151", lw=1.3, ls=(0, (5, 4)), zorder=1)
    ax.plot([SX + SW / 2, SX + SW / 2], [1.15, RET_Y], **ret)
    ax.plot([SX + SW / 2, RET_X], [RET_Y, RET_Y], **ret)
    ax.plot([RET_X, RET_X], [RET_Y, 6.97], **ret)
    _arrow(ax, (RET_X, 6.97), (CX, 6.97), color="#374151", ls="--", lw=1.3)
    ax.text((SX + SW / 2 + RET_X) / 2, RET_Y + 0.16,
            "broadcast updated global model  w_g   (the public reference for the next round)",
            fontsize=8.5, color="#374151", ha="center", style="italic")

    ax.plot([BOUND, BOUND], [1.9, 7.6], color="#9ca3af", lw=1.0, ls=(0, (4, 3)))
    ax.text(BOUND, 7.75, "network boundary", fontsize=8.5, color="#6b7280",
            ha="center", style="italic")
    ax.text(BOUND - 0.10, 1.80, "only E(Δw_i)\nand a_i cross",
            fontsize=8.5, color="#6b7280", ha="center", va="top", style="italic")

    for i, (c, lab) in enumerate([(CLIENT, "client compute"), (CRYPTO, "encrypted / crypto"),
                                  (TRUST, "trust (plaintext sketch)"), (SERVER, "server apply"),
                                  (KEYS, "separate key holder")]):
        ax.add_patch(FancyBboxPatch((0.35 + i * 2.55, 0.05), 0.28, 0.28,
                                    boxstyle="round,pad=0.02", facecolor=c, edgecolor="#333"))
        ax.text(0.72 + i * 2.55, 0.19, lab, fontsize=8.2, va="center")

    ax.set_title("HE-NeuroTrust: one federated round. The aggregation server holds only the public key and "
                 "combines ciphertexts it cannot open;\na separate authority decrypts just the aggregate, "
                 "weighted by trust scores from lightweight plaintext attestations.",
                 fontsize=10, weight="bold")
    fig.savefig(FIG / "fig1_architecture.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / "fig1_architecture.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig1_architecture")


def fig_neurofuzzy():
    from src.fuzzy.trust_engine import TrustFeatures
    from src.fuzzy.neuro_fuzzy import NeuroFuzzyTrustEngine

    rng = np.random.default_rng(0)
    def gen(n, honest):
        out = []
        for _ in range(n):
            if honest:
                out.append(TrustFeatures(float(np.clip(rng.normal(0.85, 0.12), -1, 1)),
                                         float(np.clip(rng.normal(0.15, 0.10), -1, 1)),
                                         float(rng.uniform(0.05, 0.35))))
            else:
                out.append(TrustFeatures(float(np.clip(rng.normal(-0.5, 0.25), -1, 1)),
                                         float(np.clip(rng.normal(-0.05, 0.10), -1, 1)),
                                         float(rng.uniform(0.05, 0.35))))
        return out
    hon, mal = gen(400, True), gen(400, False)
    eng = NeuroFuzzyTrustEngine(n_mf=5, lr=0.02, seed=1)
    eng.fit(hon + mal, [1.0] * 400 + [0.0] * 400, epochs=400)

    model = eng.model
    means = model.means.detach().numpy()
    spreads = np.exp(model.log_spreads.detach().numpy())
    names = ["cosine similarity", "loss improvement", "data volume"]
    lows, highs = [-1, -1, 0], [1, 1, 1]

    fig = plt.figure(figsize=(13, 4.2))
    for i in range(3):
        ax = fig.add_subplot(1, 4, i + 1)
        xs = np.linspace(lows[i], highs[i], 300)
        for k in range(model.n_mf):
            mu = np.exp(-0.5 * ((xs - means[i, k]) / (abs(spreads[i, k]) + 1e-6)) ** 2)
            ax.plot(xs, mu, lw=1.8)
            ax.fill_between(xs, mu, alpha=0.12)
        ax.set_title(f"({chr(97+i)}) learned MFs: {names[i]}", fontsize=9, weight="bold")
        ax.set_xlabel(names[i]); ax.set_ylabel("membership μ")
        ax.set_ylim(0, 1.05)

    ax = fig.add_subplot(1, 4, 4)
    gx, gy = np.meshgrid(np.linspace(-1, 1, 60), np.linspace(-1, 1, 60))
    feats = [TrustFeatures(float(a), float(b), 0.2) for a, b in zip(gx.ravel(), gy.ravel())]
    tz = eng.score_many(feats).reshape(gx.shape)
    cs = ax.contourf(gx, gy, tz, levels=20, cmap="RdYlGn", vmin=0, vmax=1)
    ax.contour(gx, gy, tz, levels=[0.40], colors="black", linewidths=1.6, linestyles="--")
    ax.set_title("(d) trust decision surface", fontsize=9, weight="bold")
    ax.set_xlabel("cosine similarity\n(data volume fixed at 0.2; dashed = τ = 0.40 gate)",
                  fontsize=8.5)
    ax.set_ylabel("loss improvement")
    fig.colorbar(cs, ax=ax, fraction=0.046, label="trust t")

    fig.suptitle("Differentiable neuro-fuzzy (ANFIS) trust engine: Gaussian membership functions are "
                 "learned from labelled attestations,\nyielding a smooth trust surface; the Zero-Trust "
                 "threshold τ carves the accept/reject boundary.",
                 fontsize=10, weight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIG / "fig2_neurofuzzy.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / "fig2_neurofuzzy.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig2_neurofuzzy")


if __name__ == "__main__":
    fig_architecture()
    fig_neurofuzzy()
    print("Section 3 figures ->", FIG)

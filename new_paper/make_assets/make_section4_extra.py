"""Supplementary Section 4 figures: confusion matrices + HMOG verification ROC/EER.

fig8_confusion.{png,pdf}  - row-normalised 8-class confusion matrices,
                            full_system_neuro (defended) vs fedavg_attack
                            (undefended), CIC-IoT-2023 seed 42.
fig9_hmog_eer.{png,pdf}   - HMOG continuous-auth verification ROC curves with
                            the Equal-Error-Rate operating point marked, for
                            full_system vs undefended FL; contextualised
                            against V-TrustFL's reported EER.
"""
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
FIG = ROOT / "new_paper" / "figures"; FIG.mkdir(parents=True, exist_ok=True)
from src.utils import load_config
from src.models import build_model

def _save(fig, n):
    fig.savefig(FIG / f"{n}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / f"{n}.pdf", bbox_inches="tight"); plt.close(fig)


def confusion_fig():
    import json
    cfg = load_config(ROOT / "configs/default.yaml")
    labels = json.load(open(ROOT / "data/processed/metadata.json"))["label_names"]
    d = np.load(ROOT / "data/processed/processed.npz")
    Xte = torch.from_numpy(d["X_test"]).float(); yte = d["y_test"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, scen, title in [(axes[0], "fedavg_attack", "(a) No defence (FedAvg + 30% attack)"),
                            (axes[1], "full_system_neuro", "(b) HE-NeuroTrust (neuro-fuzzy)")]:
        m = build_model(cfg, Xte.shape[1], len(labels)).eval()
        m.load_state_dict(torch.load(ROOT / f"results/models/{scen}_seed42.pt", map_location="cpu"))
        with torch.no_grad():
            pred = m(Xte).argmax(1).numpy()
        cm = confusion_matrix(yte, pred, labels=range(len(labels)))
        cmn = cm / np.clip(cm.sum(1, keepdims=True), 1, None)
        im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        for i in range(len(labels)):
            for j in range(len(labels)):
                if cmn[i, j] >= 0.01:
                    ax.text(j, i, f"{cmn[i,j]:.2f}", ha="center", va="center",
                            fontsize=7, color="white" if cmn[i, j] > 0.5 else "black")
        ax.set_title(title, fontsize=10, weight="bold")
        ax.set_xlabel("predicted")
        if ax is axes[0]:                      # one shared y-label, not one per panel
            ax.set_ylabel("true")
    fig.suptitle("Row-normalised confusion matrices (CIC-IoT-2023, seed 42). HE-NeuroTrust\n"
                 "substantially restores the per-class diagonal that the undefended model loses under attack.",
                 fontsize=10, weight="bold", y=1.06)
    fig.colorbar(im, ax=axes, fraction=0.025, label="recall")
    _save(fig, "fig8_confusion"); print("fig8 ok")


def hmog_eer_fig():
    cfg = load_config(ROOT / "configs/default.yaml")
    cfg.model["type"] = "cnn_lstm"; cfg.model["input_shape"] = [128, 6]
    HD = Path("D:/ZeroTrust_Gemini_2026/HMOG_ZT_Real_Project/processed_data")
    K = 20
    rng = np.random.default_rng(42)
    users = sorted(rng.choice(100, size=K, replace=False).tolist())
    Xs, ys = [], []
    for c, u in enumerate(users):
        X = np.load(HD / f"X_test_{u}.npy").astype(np.float32)
        y = np.load(HD / f"y_test_{u}.npy").reshape(-1)
        g = X[y == 1]; Xs.append(g); ys.append(np.full(len(g), c))
    X = np.concatenate(Xs); yv = np.concatenate(ys)

    fig, ax = plt.subplots(figsize=(6, 5.4))
    colors = {"fedavg": "#1f77b4", "full_system": "#2ca02c", "fedavg_attack": "#d62728"}
    titles = {"fedavg": "Clean FedAvg (no attack)",
              "full_system": "HE-NeuroTrust (under attack)",
              "fedavg_attack": "No defence (under attack)"}
    SEEDS = [42, 123, 2024, 7, 2025]      # the paper's five main-experiment seeds
    gen = np.concatenate([(yv == c).astype(int) for c in range(K)])

    def _eer_auc(ckpt):
        m = build_model(cfg, 128 * 6, K).eval()
        m.load_state_dict(torch.load(ckpt, map_location="cpu"))
        probs = []
        with torch.no_grad():
            for b in range(0, len(X), 1024):
                probs.append(torch.softmax(m(torch.from_numpy(X[b:b+1024])), 1).numpy())
        P = np.concatenate(probs)
        sco = np.concatenate([P[:, c] for c in range(K)])           # pooled one-vs-rest
        fpr, tpr, _ = roc_curve(gen, sco)
        auc = roc_auc_score(gen, sco)
        fnr = 1 - tpr
        idx = int(np.nanargmin(np.abs(fnr - fpr)))
        return fpr, tpr, auc, (fpr[idx] + fnr[idx]) / 2, idx

    for scen in ["fedavg", "full_system", "fedavg_attack"]:
        ck42 = ROOT / f"results_hmog/models/{scen}_seed42.pt"
        if not ck42.exists():
            continue
        fpr, tpr, auc, eer, idx = _eer_auc(ck42)
        eers = []
        for s in SEEDS:
            ck = ROOT / f"results_hmog/models/{scen}_seed{s}.pt"
            if ck.exists():
                eers.append(_eer_auc(ck)[3] * 100)
        m_e, s_e = float(np.mean(eers)), float(np.std(eers, ddof=1)) if len(eers) > 1 else 0.0
        ax.plot(fpr, tpr, color=colors[scen], lw=2,
                label=f"{titles[scen]}\n   seed 42: AUC={auc:.2f}, EER={eer*100:.1f}%"
                      f"  |  5 seeds: EER={m_e:.1f}±{s_e:.1f}%")
        ax.scatter([fpr[idx]], [tpr[idx]], color=colors[scen], s=40, zorder=5)
    ax.plot([0, 1], [0, 1], "k:", lw=0.8)
    ax.set_xlabel("false accept rate"); ax.set_ylabel("true accept rate")
    ax.set_title("HMOG continuous-authentication verification ROC\n"
                 "ROC from the representative seed-42 run; EER also summarised over 5 seeds",
                 fontsize=9.5, weight="bold")
    ax.legend(fontsize=7.2, loc="lower right", framealpha=0.95)
    _save(fig, "fig9_hmog_eer"); print("fig9 ok")


if __name__ == "__main__":
    confusion_fig(); hmog_eer_fig()
    print("extra figs ->", FIG)

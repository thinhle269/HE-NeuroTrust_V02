"""Quantitative privacy evaluation via gradient-inversion (DLG).

The main results table shows that the plaintext robust defences (Krum,
FoolsGold, Bulyan, ...) match our system on accuracy.  Their hidden cost -
and our system's unique advantage - is *privacy*: those defences require
the server to see each client's raw gradient, which is directly invertible
back to the client's training data.  This script measures that leakage.

We implement the Deep-Leakage-from-Gradients attack (Zhu et al., NeurIPS
2019): given a client's gradient and the shared model, an honest-but-
curious server optimises a dummy input/label pair to reproduce the
observed gradient, thereby reconstructing the client's private feature
vector.  We compare three observability regimes:

* ``plaintext_single``  - server sees a single-sample gradient
                          (the worst case for the plaintext baselines).
* ``plaintext_batch``   - server sees a B-sample batch-averaged gradient
                          (what a real FedAvg round exposes per client).
* ``he_aggregate``      - server sees only the Paillier-decrypted *sum*
                          over all clients (our full_system): individual
                          gradients are never exposed, so the attacker must
                          invert an aggregate of N*B samples and fails.

Metric: reconstruction MSE and cosine similarity between the recovered and
true standardised feature vectors (lower MSE / higher cosine = more
leakage).  The output table + bar chart make the privacy gap explicit.

Usage
-----
    python run_privacy_eval.py                 # default: 20 targets
    python run_privacy_eval.py --n-targets 40 --dlg-iters 300
"""
from __future__ import annotations

import matplotlib                                              # noqa: E402
matplotlib.use("Agg")                                          # noqa: E402

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn

from src.data import DataPreprocessor
from src.models import build_model
from src.utils import get_logger, load_config, set_global_seed

PROJECT_ROOT = Path(__file__).resolve().parent
OUT = PROJECT_ROOT / "results" / "privacy_eval"


def _true_gradient(model, loss_fn, X, y):
    """Flat gradient of the loss at (X, y) w.r.t. all parameters."""
    model.zero_grad(set_to_none=True)
    out = model(X)
    loss = loss_fn(out, y)
    grads = torch.autograd.grad(loss, list(model.parameters()), create_graph=False)
    return [g.detach().clone() for g in grads]


def dlg_attack(model, loss_fn, target_grad, n_features, num_classes,
               iters=300, lr=0.1, device="cpu", seed=0):
    """Reconstruct a single input from ``target_grad`` (DLG).

    Returns the recovered feature vector (n_features,).
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    dummy_x = torch.randn(1, n_features, generator=g).to(device).requires_grad_(True)
    dummy_y = torch.randn(1, num_classes, generator=g).to(device).requires_grad_(True)
    opt = torch.optim.LBFGS([dummy_x, dummy_y], lr=lr, max_iter=20)

    def closure():
        opt.zero_grad(set_to_none=True)
        out = model(dummy_x)
        logp = torch.log_softmax(out, dim=1)
        loss = -(torch.softmax(dummy_y, dim=1) * logp).sum(dim=1).mean()
        grad = torch.autograd.grad(loss, list(model.parameters()), create_graph=True)
        diff = sum(((gx - gt) ** 2).sum() for gx, gt in zip(grad, target_grad))
        diff.backward()
        return diff

    for _ in range(iters // 20):
        opt.step(closure)
    return dummy_x.detach().cpu().numpy().reshape(-1)


def _recon_metrics(true_vec, rec_vec):
    """Reconstruction quality of one DLG trial.

    Convergence criterion: a trial counts as *valid* only if the optimiser
    returned an all-finite reconstruction with a non-degenerate norm.  L-BFGS
    occasionally diverges to NaN/Inf (or collapses to the zero vector), in which
    case no similarity is defined; such trials are reported as NaN and are
    excluded from the summary statistics rather than being silently coerced to
    a number.  The per-regime count of valid trials is written to
    ``privacy_summary.csv`` as ``n_valid``.
    """
    if not np.all(np.isfinite(rec_vec)) or np.linalg.norm(rec_vec) < 1e-8:
        return float("nan"), float("nan")
    mse = float(np.mean((true_vec - rec_vec) ** 2))
    a = true_vec / (np.linalg.norm(true_vec) + 1e-9)
    b = rec_vec / (np.linalg.norm(rec_vec) + 1e-9)
    cos = float(np.dot(a, b))
    return mse, cos


def _bootstrap_ci(values, n_boot=2000, alpha=0.05, seed=0):
    """Percentile bootstrap CI of the mean over the valid trials."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan"), 0
    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(v, v.size, replace=True).mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi), int(v.size)


def parse_args():
    p = argparse.ArgumentParser(description="Gradient-inversion privacy eval")
    p.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    p.add_argument("--n-targets", type=int, default=20)
    p.add_argument("--dlg-iters", type=int, default=300)
    p.add_argument("--batch", type=int, default=8, help="batch size for plaintext_batch")
    p.add_argument("--he-clients", type=int, default=10, help="clients summed in he_aggregate")
    return p.parse_args()


def main():
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    logger = get_logger("privacy_eval", OUT / "logs")
    cfg = load_config(args.config)
    set_global_seed(cfg.seed)
    device = torch.device("cpu")

    split = DataPreprocessor(cfg, PROJECT_ROOT).run(force=False)
    model = build_model(cfg, split.num_features, split.num_classes).to(device).eval()
    loss_fn = nn.CrossEntropyLoss()
    Xte = torch.from_numpy(split.X_test).float()
    yte = torch.from_numpy(split.y_test).long()
    rng = np.random.default_rng(cfg.seed)

    rows = []
    n = int(args.n_targets)
    for t in range(n):
        idx = int(rng.integers(0, len(yte)))
        x1 = Xte[idx:idx + 1].to(device)
        y1 = yte[idx:idx + 1].to(device)
        true_vec = x1.cpu().numpy().reshape(-1)

        g_single = _true_gradient(model, loss_fn, x1, y1)
        rec = dlg_attack(model, loss_fn, g_single, split.num_features,
                         split.num_classes, iters=args.dlg_iters, device=device, seed=t)
        mse, cos = _recon_metrics(true_vec, rec)
        rows.append({"regime": "plaintext_single", "target": t, "mse": mse, "cosine": cos})

        bidx = rng.integers(0, len(yte), size=args.batch)
        Xb = Xte[bidx].to(device); yb = yte[bidx].to(device)
        g_batch = _true_gradient(model, loss_fn, Xb, yb)
        rec_b = dlg_attack(model, loss_fn, g_batch, split.num_features,
                           split.num_classes, iters=args.dlg_iters, device=device, seed=1000 + t)
        mse_b, cos_b = _recon_metrics(Xb[0].cpu().numpy().reshape(-1), rec_b)
        rows.append({"regime": "plaintext_batch", "target": t, "mse": mse_b, "cosine": cos_b})

        agg = [torch.zeros_like(g) for g in g_batch]
        for _c in range(args.he_clients):
            cidx = rng.integers(0, len(yte), size=args.batch)
            Xc = Xte[cidx].to(device); yc = yte[cidx].to(device)
            if _c == 0:
                Xc = Xc.clone(); yc = yc.clone()
                Xc[0] = x1[0]; yc[0] = y1[0]      # inject the victim as a participant
            gc = _true_gradient(model, loss_fn, Xc, yc)
            for a, gg in zip(agg, gc):
                a += gg
        rec_h = dlg_attack(model, loss_fn, agg, split.num_features,
                           split.num_classes, iters=args.dlg_iters, device=device, seed=2000 + t)
        mse_h, cos_h = _recon_metrics(true_vec, rec_h)
        rows.append({"regime": "he_aggregate", "target": t, "mse": mse_h, "cosine": cos_h})

        if (t + 1) % 5 == 0:
            logger.info("processed %d/%d targets", t + 1, n)

    raw = pd.DataFrame(rows)
    raw.to_csv(OUT / "privacy_raw.csv", index=False)
    summary = raw.groupby("regime")[["mse", "cosine"]].agg(["mean", "std"]).round(4)
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index()
    ci_rows = []
    for reg in summary["regime"]:
        vals = raw.loc[raw.regime == reg, "cosine"]
        lo, hi, n_valid = _bootstrap_ci(vals)
        ci_rows.append({"regime": reg, "cosine_ci_lo": round(lo, 4),
                        "cosine_ci_hi": round(hi, 4), "n_valid": n_valid,
                        "n_trials": int(len(vals))})
    summary = summary.merge(pd.DataFrame(ci_rows), on="regime")
    summary.to_csv(OUT / "privacy_summary.csv", index=False)
    logger.info("Privacy summary (lower MSE / higher cosine = more leakage):\n%s",
                summary.to_string(index=False))

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    order = ["plaintext_single", "plaintext_batch", "he_aggregate"]
    sub = summary.set_index("regime").reindex(order).reset_index()
    sns.barplot(data=sub, x="regime", y="cosine_mean", ax=axes[0],
                palette=["#d62728", "#ff7f0e", "#2ca02c"], edgecolor="black")
    axes[0].errorbar(range(len(sub)), sub["cosine_mean"], yerr=sub["cosine_std"],
                     fmt="none", ecolor="black", capsize=4)
    axes[0].set_title("Reconstruction cosine similarity (higher = more leakage)")
    axes[0].set_ylabel("cosine(true, recovered)"); axes[0].set_xlabel("")
    sns.barplot(data=sub, x="regime", y="mse_mean", ax=axes[1],
                palette=["#d62728", "#ff7f0e", "#2ca02c"], edgecolor="black")
    axes[1].errorbar(range(len(sub)), sub["mse_mean"], yerr=sub["mse_std"],
                     fmt="none", ecolor="black", capsize=4)
    axes[1].set_title("Reconstruction MSE (lower = more leakage)")
    axes[1].set_ylabel("MSE(true, recovered)"); axes[1].set_xlabel("")
    for ax in axes:
        ax.tick_params(axis="x", rotation=15)
    fig.suptitle("Gradient-inversion (DLG) leakage: plaintext defences vs. HE aggregation")
    fig.savefig(OUT / "privacy_eval.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "privacy_eval.pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info("Privacy eval done. Outputs under %s", OUT)


if __name__ == "__main__":
    main()

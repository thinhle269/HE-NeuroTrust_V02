"""HMOG verification metrics (EER / AUC) for direct comparison with V-TrustFL.

V-TrustFL reports continuous-authentication *verification* performance on
HMOG (EER = 8.99%, AUC = 0.9675).  Our paper trains a K-user *identification*
model, but the two are reconcilable: for user ``u`` (class ``c``) the
verification score of a window is the model's softmax probability of class
``c``.  Genuine windows (user u's own) should score high; impostor windows
(any other user's) low.  We compute per-user EER and AUC from that score and
average over users, yielding numbers directly comparable to V-TrustFL - on
the same dataset, from the trained models this paper already produced.

We evaluate the clean ``fedavg`` model (closest to V-TrustFL's non-private,
attack-free setting) and the security-hardened ``full_system`` /
``full_system_neuro`` models (which additionally provide Paillier privacy
and Byzantine robustness that V-TrustFL does not).

Usage
-----
    python run_hmog_verification.py
    python run_hmog_verification.py --scenarios fedavg full_system --n-users 20
"""
from __future__ import annotations

import matplotlib                                              # noqa: E402
matplotlib.use("Agg")                                          # noqa: E402

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, roc_curve

from src.models import build_model
from src.utils import load_config, get_logger

PROJECT_ROOT = Path(__file__).resolve().parent
HMOG_DIR = Path("D:/ZeroTrust_Gemini_2026/HMOG_ZT_Real_Project/processed_data")
MODELS = PROJECT_ROOT / "results_hmog" / "models"
OUT = PROJECT_ROOT / "results_hmog"

VTRUSTFL = {"EER_pct": 8.99, "AUC": 0.9675}


def sampled_users(seed: int, n_users: int):
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(100, size=n_users, replace=False).tolist())


def load_test(users):
    """K-class identification test set: genuine windows per user -> class."""
    Xs, ys = [], []
    for cls, u in enumerate(users):
        X = np.load(HMOG_DIR / f"X_test_{u}.npy").astype(np.float32)
        y = np.load(HMOG_DIR / f"y_test_{u}.npy").reshape(-1)
        g = X[y == 1]
        Xs.append(g); ys.append(np.full(len(g), cls, dtype=np.int64))
    return np.concatenate(Xs), np.concatenate(ys)


def eer_from_scores(y_true, scores):
    """Equal Error Rate: point where FPR == FNR."""
    fpr, tpr, _ = roc_curve(y_true, scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[idx] + fnr[idx]) / 2.0)


def verify_model(model, X, y, n_users, device):
    """Return mean EER, mean AUC over the n_users one-vs-rest verifications."""
    model.eval()
    probs = []
    with torch.no_grad():
        for b in range(0, len(X), 1024):
            xb = torch.from_numpy(X[b:b + 1024]).to(device)
            probs.append(torch.softmax(model(xb), dim=1).cpu().numpy())
    P = np.concatenate(probs, axis=0)          # (N, n_users)
    eers, aucs = [], []
    for c in range(n_users):
        genuine = (y == c).astype(int)          # 1 = user c, 0 = impostor
        score = P[:, c]
        if genuine.sum() == 0 or genuine.sum() == len(genuine):
            continue
        aucs.append(float(roc_auc_score(genuine, score)))
        eers.append(eer_from_scores(genuine, score))
    return float(np.mean(eers)), float(np.mean(aucs))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    ap.add_argument("--scenarios", nargs="+",
                    default=["fedavg", "fedavg_attack", "full_system", "full_system_neuro"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2024, 7, 2025])
    ap.add_argument("--n-users", type=int, default=20)
    return ap.parse_args()


def main():
    args = parse_args()
    logger = get_logger("hmog_verify", OUT / "logs")
    cfg = load_config(args.config)
    cfg.model["type"] = "cnn_lstm"
    cfg.model["input_shape"] = [128, 6]
    device = torch.device("cpu")

    rows = []
    for seed in args.seeds:
        users = sampled_users(42, args.n_users)
        X, y = load_test(users)
        for scen in args.scenarios:
            ckpt = MODELS / f"{scen}_seed{seed}.pt"
            if not ckpt.exists():
                logger.warning("missing %s", ckpt); continue
            model = build_model(cfg, in_features=128 * 6, num_classes=args.n_users).to(device)
            model.load_state_dict(torch.load(ckpt, map_location=device))
            eer, auc = verify_model(model, X, y, args.n_users, device)
            rows.append({"scenario": scen, "seed": seed,
                         "EER_pct": round(100 * eer, 2), "AUC": round(auc, 4)})
            logger.info("[%s seed=%d] EER=%.2f%% AUC=%.4f", scen, seed, 100 * eer, auc)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "csv" / "hmog_verification_per_seed.csv", index=False)
    summ = df.groupby("scenario")[["EER_pct", "AUC"]].agg(["mean", "std"]).round(3)
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ = summ.reset_index()
    summ.to_csv(OUT / "csv" / "hmog_verification_summary.csv", index=False)

    print("\n================ HMOG VERIFICATION (EER / AUC) ================")
    print(summ.to_string(index=False))
    print(f"\nReference V-TrustFL (Le et al., clean, non-private): "
          f"EER={VTRUSTFL['EER_pct']}%  AUC={VTRUSTFL['AUC']}")
    print("Note: our identification-derived verification is a closed-set,\n"
          "compact-model (2146-param) setting under 30% Byzantine attack + HE;\n"
          "the security layer (privacy + robustness) is the paper's contribution.")
    logger.info("Verification eval done. Outputs under %s/csv", OUT)


if __name__ == "__main__":
    main()

"""HMOG continuous-authentication loader (user-identification framing).

The HMOG dataset (used by the authors' V-TrustFL work) ships as 100
pre-split users, each with ``X_{train,val,test}_i.npy`` windows of shape
``(N, 128, 6)`` - 128 timesteps of 6 inertial channels - and balanced
genuine/impostor labels.

Per-user genuine/impostor *verification* is not poolable into a single
shared federated model (each user's "genuine" class is different), so we
adopt the standard behavioural-biometric *identification* framing: each
sampled user contributes their **genuine** windows as one class, and the
shared model learns to identify which user produced a given inertial
window.  This is directly analogous to the keystroke-dynamics
identification task and fits the same FL pipeline (multiclass, shared
model, Dirichlet partition), while exercising the sequence-model
(1D-CNN-LSTM) path.

The original per-user train/val/test partition is preserved (no temporal
leakage): train windows go to train, etc.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.preprocessing import LabelEncoder

from .preprocessor import ProcessedSplit
from ..utils.logger import get_logger

HMOG_DIR_DEFAULT = "D:/ZeroTrust_Gemini_2026/HMOG_ZT_Real_Project/processed_data"


class HmogPreprocessor:
    """Load ``K`` HMOG users as a K-class identification ProcessedSplit."""

    def __init__(self, cfg, project_root: Path):
        self.cfg = cfg
        self.project_root = Path(project_root)
        d = cfg.data
        self.dir = Path(d.get("hmog_dir", HMOG_DIR_DEFAULT) if hasattr(d, "get")
                        else HMOG_DIR_DEFAULT)
        self.n_users = int(d.get("hmog_users", 20)) if hasattr(d, "get") else 20
        self.logger = get_logger("data.hmog", self.project_root / cfg.paths.logs_dir,
                                 level=cfg.logging.level)
        self.partition_key_ = None

    def _load(self, split, users):
        Xs, ys = [], []
        for cls, u in enumerate(users):
            X = np.load(self.dir / f"X_{split}_{u}.npy").astype(np.float32)
            y = np.load(self.dir / f"y_{split}_{u}.npy").reshape(-1)
            genuine = X[y == 1]                       # user u's authentic windows
            Xs.append(genuine)
            ys.append(np.full(len(genuine), cls, dtype=np.int64))
        return np.concatenate(Xs), np.concatenate(ys)

    def run(self, force: bool = False) -> ProcessedSplit:
        rng = np.random.default_rng(self.cfg.seed)
        users = sorted(rng.choice(100, size=self.n_users, replace=False).tolist())
        X_train, y_train = self._load("train", users)
        X_val, y_val = self._load("val", users)
        X_test, y_test = self._load("test", users)

        perm = rng.permutation(len(y_train))
        X_train, y_train = X_train[perm], y_train[perm]

        encoder = LabelEncoder().fit(np.arange(self.n_users))
        label_names = [f"user{u}" for u in users]
        feature_names = [f"t{t}_c{c}" for t in range(X_train.shape[1])
                         for c in range(X_train.shape[2])]

        self.logger.info("HMOG identification: %d users | train=%d val=%d test=%d "
                         "window=%s", self.n_users, len(y_train), len(y_val),
                         len(y_test), X_train.shape[1:])
        self.partition_key_ = y_train.copy()
        return ProcessedSplit(
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            X_test=X_test, y_test=y_test,
            feature_names=feature_names,
            label_names=label_names,
            label_encoder=encoder,
            scaler=None,
        )

"""Generic loader for the standardised ``{name}.npz`` + ``{name}.meta.json``
datasets, enabling multi-dataset evaluation of the FL-IDS pipeline.

The sibling HE-FedSec project ships three pre-processed datasets in a clean,
uniform format that plugs directly into this pipeline:

* ``edgeiiot``   - Edge-IIoTset intrusion detection (15 classes, 34 features).
* ``keystroke``  - CMU keystroke-dynamics behavioural biometric
                   (51 subjects, 31 features) - a continuous-authentication
                   analogue of HMOG.
* ``rba``        - Risk-Based Authentication (binary, 14 features).

Each ``.npz`` holds three aligned arrays:

* ``X``              (N, D) float32 feature matrix,
* ``y``              (N,)   int64 class labels,
* ``partition_key``  (N,)   int64 natural owner id (subject / source) - used
                     for a realistic *by-owner* federated partition, which is
                     the authentic setup for the authentication datasets.

The loader mirrors :class:`DataPreprocessor` and returns the identical
:class:`ProcessedSplit` dataclass, so the rest of the pipeline
(partitioner, server, evaluation) is dataset-agnostic.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler, StandardScaler

from .preprocessor import ProcessedSplit
from ..utils.logger import get_logger


def _make_scaler(name: str):
    name = (name or "standard").lower()
    return {"standard": StandardScaler, "minmax": MinMaxScaler,
            "robust": RobustScaler}.get(name, StandardScaler)()


class NpzPreprocessor:
    """Load a ``{name}.npz`` dataset into a train/val/test ProcessedSplit."""

    def __init__(self, cfg, project_root: Path):
        self.cfg = cfg
        self.project_root = Path(project_root)
        d = cfg.data
        self.npz_path = Path(d.npz_path)
        self.name = str(d.get("npz_name", self.npz_path.stem)) if hasattr(d, "get") \
            else self.npz_path.stem
        self.logger = get_logger("data.npz", self.project_root / cfg.paths.logs_dir,
                                 level=cfg.logging.level)
        self.partition_key_: Optional[np.ndarray] = None  # aligned to y_train

    def run(self, force: bool = False) -> ProcessedSplit:
        meta_path = self.npz_path.with_suffix(".meta.json")
        meta = {}
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        data = np.load(self.npz_path, allow_pickle=True)
        X = np.asarray(data["X"], dtype=np.float32)
        y = np.asarray(data["y"]).astype(np.int64)
        pkey = np.asarray(data["partition_key"]).astype(np.int64) \
            if "partition_key" in data else np.zeros(len(y), dtype=np.int64)
        self.logger.info("Loaded %s: X=%s y=%s (%d classes)",
                         self.name, X.shape, y.shape, len(np.unique(y)))

        finite = np.isfinite(X).all(axis=1)
        if not finite.all():
            self.logger.info("Dropping %d non-finite rows", int((~finite).sum()))
            X, y, pkey = X[finite], y[finite], pkey[finite]

        cap = int(getattr(self.cfg.data, "max_rows_per_class", 0) or 0)
        if cap > 0:
            X, y, pkey = self._cap_per_class(X, y, pkey, cap)

        encoder = LabelEncoder()
        y_enc = encoder.fit_transform(y).astype(np.int64)
        class_names = meta.get("class_names")
        label_names = [str(c) for c in class_names] if class_names and \
            len(class_names) == len(encoder.classes_) else \
            [str(c) for c in encoder.classes_]
        feature_names = meta.get("feature_names") or [f"f{i}" for i in range(X.shape[1])]

        idx = np.arange(len(y_enc))
        test_size = float(self.cfg.data.test_size)
        val_size = float(self.cfg.data.val_size)
        strat = y_enc if np.min(np.bincount(y_enc)) >= 2 else None
        tr_idx, te_idx = train_test_split(idx, test_size=test_size,
                                          stratify=strat, random_state=self.cfg.seed)
        strat_tr = y_enc[tr_idx] if strat is not None else None
        val_rel = val_size / (1.0 - test_size)
        tr_idx, va_idx = train_test_split(tr_idx, test_size=val_rel,
                                          stratify=strat_tr, random_state=self.cfg.seed)

        scaler = _make_scaler(self.cfg.data.scaler)
        X_train = scaler.fit_transform(X[tr_idx]).astype(np.float32)
        X_val = scaler.transform(X[va_idx]).astype(np.float32)
        X_test = scaler.transform(X[te_idx]).astype(np.float32)

        self.partition_key_ = pkey[tr_idx]

        split = ProcessedSplit(
            X_train=X_train, y_train=y_enc[tr_idx],
            X_val=X_val,     y_val=y_enc[va_idx],
            X_test=X_test,   y_test=y_enc[te_idx],
            feature_names=list(feature_names),
            label_names=list(label_names),
            label_encoder=encoder,
            scaler=scaler,
        )
        self.logger.info("%s ready: train=%d val=%d test=%d features=%d classes=%d",
                         self.name, len(split.y_train), len(split.y_val),
                         len(split.y_test), split.num_features, split.num_classes)
        return split

    def _cap_per_class(self, X, y, pkey, cap):
        rng = np.random.default_rng(self.cfg.seed)
        keep = []
        for c in np.unique(y):
            ci = np.where(y == c)[0]
            if len(ci) > cap:
                ci = rng.choice(ci, size=cap, replace=False)
            keep.append(ci)
        keep = np.concatenate(keep)
        rng.shuffle(keep)
        return X[keep], y[keep], pkey[keep]

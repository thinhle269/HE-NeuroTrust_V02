"""CIC-IoT-2023 preprocessing pipeline.

The raw dataset ships as per-attack CSV folders plus pre-merged sample files
(``Merged0*.csv``).  We work from the merged files because they already provide
a balanced random sample across attack types while staying small enough for
single-machine experimentation (~580 MB total, ~2.83M flows).

Pipeline (executed once, then cached):

    1. Concatenate the merged CSVs.
    2. Sanity: drop duplicate rows, rows with NaN/Inf in feature columns.
    3. Normalise labels: upper-case, strip whitespace.
    4. Optionally collapse the 34 attack types into 8 super-classes (see
       :data:`LABEL_GROUPS`).
    5. Stratified down-sampling per class (``max_rows_per_class``) to keep the
       training pipeline tractable while preserving class diversity.
    6. Drop constant or near-perfectly-correlated features.
    7. Fit a feature scaler on the *train* split only and apply to all splits.
    8. Persist X/y train/val/test as compressed ``.npz`` plus the metadata
       (label encoder, scaler, feature list, label distribution).

The persisted artefacts live under ``data/processed`` and are reused on
subsequent runs unless ``force=True`` or the cache version differs.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler, StandardScaler

from ..utils.logger import get_logger


LABEL_GROUPS: Dict[str, str] = {
    "BENIGN": "Benign",
    "BENIGNTRAFFIC": "Benign",
    "DDOS-ACK_FRAGMENTATION": "DDoS",
    "DDOS-HTTP_FLOOD": "DDoS",
    "DDOS-ICMP_FLOOD": "DDoS",
    "DDOS-ICMP_FRAGMENTATION": "DDoS",
    "DDOS-PSHACK_FLOOD": "DDoS",
    "DDOS-RSTFINFLOOD": "DDoS",
    "DDOS-SYN_FLOOD": "DDoS",
    "DDOS-SLOWLORIS": "DDoS",
    "DDOS-SYNONYMOUSIP_FLOOD": "DDoS",
    "DDOS-TCP_FLOOD": "DDoS",
    "DDOS-UDP_FLOOD": "DDoS",
    "DDOS-UDP_FRAGMENTATION": "DDoS",
    "DOS-HTTP_FLOOD": "DoS",
    "DOS-SYN_FLOOD": "DoS",
    "DOS-TCP_FLOOD": "DoS",
    "DOS-UDP_FLOOD": "DoS",
    "MIRAI-GREETH_FLOOD": "Mirai",
    "MIRAI-GREIP_FLOOD": "Mirai",
    "MIRAI-UDPPLAIN": "Mirai",
    "RECON-HOSTDISCOVERY": "Recon",
    "RECON-OSSCAN": "Recon",
    "RECON-PINGSWEEP": "Recon",
    "RECON-PORTSCAN": "Recon",
    "VULNERABILITYSCAN": "Recon",
    "BACKDOOR_MALWARE": "Web",
    "BROWSERHIJACKING": "Web",
    "COMMANDINJECTION": "Web",
    "SQLINJECTION": "Web",
    "UPLOADING_ATTACK": "Web",
    "XSS": "Web",
    "DICTIONARYBRUTEFORCE": "BruteForce",
    "DNS_SPOOFING": "Spoofing",
    "MITM-ARPSPOOFING": "Spoofing",
}


_CACHE_VERSION = "1.0.0"


@dataclass
class ProcessedSplit:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]
    label_names: List[str]                # ordered to match encoder
    label_encoder: LabelEncoder
    scaler: object

    @property
    def num_features(self) -> int:
        return self.X_train.shape[1]

    @property
    def num_classes(self) -> int:
        return len(self.label_names)


class DataPreprocessor:
    """Build (or load cached) train/val/test splits for CIC-IoT-2023."""

    def __init__(self, cfg, project_root: Path):
        self.cfg = cfg
        self.project_root = Path(project_root)
        self.raw_dir = Path(cfg.paths.raw_csv_dir)
        self.processed_dir = self.project_root / cfg.paths.processed_dir
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.results_csv_dir = self.project_root / cfg.paths.csv_dir
        self.results_csv_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger("data.preprocess", self.project_root / cfg.paths.logs_dir,
                                 level=cfg.logging.level)
        self._cache_path = self.processed_dir / "processed.npz"
        self._meta_path = self.processed_dir / "metadata.json"
        self._scaler_path = self.processed_dir / "scaler.joblib"
        self._encoder_path = self.processed_dir / "label_encoder.joblib"

    def run(self, force: bool = False) -> ProcessedSplit:
        if not force and self._cache_valid():
            self.logger.info("Loading cached processed dataset from %s", self._cache_path)
            return self._load_cache()

        self.logger.info("Building processed dataset from raw CSVs (force=%s)", force)
        df = self._load_raw()
        df = self._clean(df)
        df = self._group_labels(df)
        df = self._stratified_subsample(df)
        self._dump_class_distribution(df, "after_subsample")

        feature_cols = [c for c in df.columns if c != "Label"]
        X = df[feature_cols].to_numpy(dtype=np.float32)
        y_raw = df["Label"].to_numpy()

        encoder = LabelEncoder()
        y = encoder.fit_transform(y_raw).astype(np.int64)

        keep_mask, feature_cols = self._drop_constant_and_correlated(X, feature_cols)
        X = X[:, keep_mask]

        X_tmp, X_test, y_tmp, y_test = train_test_split(
            X, y, test_size=self.cfg.data.test_size, stratify=y,
            random_state=self.cfg.seed,
        )
        val_relative = self.cfg.data.val_size / (1.0 - self.cfg.data.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_tmp, y_tmp, test_size=val_relative, stratify=y_tmp,
            random_state=self.cfg.seed,
        )
        del X, y, X_tmp, y_tmp

        scaler = self._make_scaler(self.cfg.data.scaler)
        X_train = scaler.fit_transform(X_train).astype(np.float32)
        X_val = scaler.transform(X_val).astype(np.float32)
        X_test = scaler.transform(X_test).astype(np.float32)

        split = ProcessedSplit(
            X_train=X_train, y_train=y_train,
            X_val=X_val,     y_val=y_val,
            X_test=X_test,   y_test=y_test,
            feature_names=feature_cols,
            label_names=list(encoder.classes_),
            label_encoder=encoder,
            scaler=scaler,
        )
        self._save_cache(split)
        self.logger.info("Saved processed dataset cache to %s", self._cache_path)
        return split

    def _cache_valid(self) -> bool:
        if not (self._cache_path.exists() and self._meta_path.exists()
                and self._scaler_path.exists() and self._encoder_path.exists()):
            return False
        try:
            meta = json.loads(self._meta_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        if meta.get("version") != _CACHE_VERSION:
            return False
        for key in ("task", "max_rows_per_class", "test_size", "val_size", "scaler",
                    "group_labels", "drop_constant", "drop_correlated_threshold"):
            if meta.get("config", {}).get(key) != getattr(self.cfg.data, key):
                return False
        if meta.get("seed") != self.cfg.seed:
            return False
        return True

    def _save_cache(self, split: ProcessedSplit) -> None:
        np.savez_compressed(
            self._cache_path,
            X_train=split.X_train, y_train=split.y_train,
            X_val=split.X_val,     y_val=split.y_val,
            X_test=split.X_test,   y_test=split.y_test,
        )
        joblib.dump(split.scaler, self._scaler_path)
        joblib.dump(split.label_encoder, self._encoder_path)
        meta = {
            "version": _CACHE_VERSION,
            "seed": self.cfg.seed,
            "config": {
                "task": self.cfg.data.task,
                "max_rows_per_class": self.cfg.data.max_rows_per_class,
                "test_size": self.cfg.data.test_size,
                "val_size": self.cfg.data.val_size,
                "scaler": self.cfg.data.scaler,
                "group_labels": self.cfg.data.group_labels,
                "drop_constant": self.cfg.data.drop_constant,
                "drop_correlated_threshold": self.cfg.data.drop_correlated_threshold,
            },
            "feature_names": split.feature_names,
            "label_names": split.label_names,
            "shape": {
                "train": list(split.X_train.shape),
                "val": list(split.X_val.shape),
                "test": list(split.X_test.shape),
            },
        }
        self._meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def _load_cache(self) -> ProcessedSplit:
        with np.load(self._cache_path) as data:
            X_train = data["X_train"]; y_train = data["y_train"]
            X_val = data["X_val"];     y_val = data["y_val"]
            X_test = data["X_test"];   y_test = data["y_test"]
        scaler = joblib.load(self._scaler_path)
        encoder = joblib.load(self._encoder_path)
        meta = json.loads(self._meta_path.read_text(encoding="utf-8"))
        return ProcessedSplit(
            X_train=X_train, y_train=y_train,
            X_val=X_val,     y_val=y_val,
            X_test=X_test,   y_test=y_test,
            feature_names=meta["feature_names"],
            label_names=meta["label_names"],
            label_encoder=encoder,
            scaler=scaler,
        )

    def _load_raw(self) -> pd.DataFrame:
        frames = []
        for fname in self.cfg.paths.raw_merged_files:
            fpath = self.raw_dir / fname
            if not fpath.exists():
                self.logger.warning("Missing merged file %s, skipping", fpath)
                continue
            self.logger.info("Loading %s", fpath)
            frames.append(pd.read_csv(fpath, low_memory=False))
        if not frames:
            raise FileNotFoundError(
                f"No merged CSVs found in {self.raw_dir}. "
                f"Check configs/default.yaml -> paths.raw_csv_dir.")
        df = pd.concat(frames, ignore_index=True)
        self.logger.info("Loaded %d rows / %d columns from raw", len(df), df.shape[1])
        return df

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        if "Label" not in df.columns:
            raise KeyError("Raw data is missing 'Label' column")
        df["Label"] = df["Label"].astype(str).str.strip().str.upper()
        df = df.replace([np.inf, -np.inf], np.nan)
        before = len(df)
        df = df.dropna()
        df = df.drop_duplicates()
        self.logger.info("Dropped %d rows after dedup+NaN cleaning -> %d remain",
                         before - len(df), len(df))
        return df

    def _group_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.cfg.data.task == "binary":
            df = df.copy()
            df["Label"] = np.where(df["Label"].str.contains("BENIGN"), "Benign", "Attack")
            return df
        if not self.cfg.data.group_labels:
            return df
        unmapped = sorted(set(df["Label"].unique()) - set(LABEL_GROUPS.keys()))
        if unmapped:
            self.logger.warning("Unmapped attack labels (kept as-is): %s", unmapped)
        df = df.copy()
        df["Label"] = df["Label"].map(lambda x: LABEL_GROUPS.get(x, x.title()))
        return df

    def _stratified_subsample(self, df: pd.DataFrame) -> pd.DataFrame:
        cap = self.cfg.data.max_rows_per_class
        if cap is None or cap <= 0:
            return df
        rng = np.random.default_rng(self.cfg.seed)
        parts = []
        for label, group in df.groupby("Label", sort=False):
            if len(group) > cap:
                idx = rng.choice(len(group), size=cap, replace=False)
                parts.append(group.iloc[idx])
            else:
                parts.append(group)
        out = pd.concat(parts, ignore_index=True)
        out = out.sample(frac=1.0, random_state=self.cfg.seed).reset_index(drop=True)
        self.logger.info("After stratified cap=%d -> %d rows over %d classes",
                         cap, len(out), out["Label"].nunique())
        return out

    def _drop_constant_and_correlated(
        self, X: np.ndarray, feature_cols: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        keep = np.ones(X.shape[1], dtype=bool)
        if self.cfg.data.drop_constant:
            stds = X.std(axis=0)
            const_mask = stds < 1e-8
            keep &= ~const_mask
            if const_mask.any():
                self.logger.info("Dropping %d constant features: %s",
                                 const_mask.sum(),
                                 [feature_cols[i] for i in np.where(const_mask)[0]])

        thr = self.cfg.data.drop_correlated_threshold
        if thr is not None and 0 < thr < 1.0:
            kept_idx = np.where(keep)[0]
            if len(kept_idx) > 1:
                sub = X[:, kept_idx]
                corr = np.corrcoef(sub, rowvar=False)
                corr = np.nan_to_num(corr)
                drop_local: set = set()
                for i in range(corr.shape[0]):
                    if i in drop_local:
                        continue
                    for j in range(i + 1, corr.shape[1]):
                        if j in drop_local:
                            continue
                        if abs(corr[i, j]) >= thr:
                            drop_local.add(j)
                if drop_local:
                    drop_global = kept_idx[list(drop_local)]
                    keep[drop_global] = False
                    self.logger.info("Dropping %d highly-correlated features (|r|>=%.2f): %s",
                                     len(drop_local), thr,
                                     [feature_cols[i] for i in drop_global])
        new_cols = [feature_cols[i] for i in np.where(keep)[0]]
        return keep, new_cols

    @staticmethod
    def _make_scaler(name: str):
        name = name.lower()
        if name == "standard":
            return StandardScaler()
        if name == "minmax":
            return MinMaxScaler()
        if name == "robust":
            return RobustScaler()
        raise ValueError(f"Unknown scaler: {name}")

    def _dump_class_distribution(self, df: pd.DataFrame, suffix: str) -> None:
        dist = df["Label"].value_counts().rename_axis("Label").reset_index(name="count")
        out = self.results_csv_dir / f"class_distribution_{suffix}.csv"
        dist.to_csv(out, index=False)
        self.logger.info("Wrote class distribution -> %s", out)

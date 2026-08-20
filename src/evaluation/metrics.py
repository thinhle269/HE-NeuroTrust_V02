"""Classification metrics used by both the centralized baseline and FL."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    precision_score, recall_score,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int,
                    loss: Optional[float] = None,
                    with_confusion: bool = False) -> Dict:
    if len(y_true) == 0:
        out = {"accuracy": 0.0, "macro_f1": 0.0, "weighted_f1": 0.0,
               "precision_macro": 0.0, "recall_macro": 0.0}
        if loss is not None:
            out["loss"] = float(loss)
        return out

    labels = list(range(num_classes))
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro",
                                   zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted",
                                      zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, labels=labels,
                                                 average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, labels=labels,
                                           average="macro", zero_division=0)),
    }
    if loss is not None:
        out["loss"] = float(loss)
    if with_confusion:
        out["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=labels)
        out["classification_report"] = classification_report(
            y_true, y_pred, labels=labels, output_dict=True, zero_division=0,
        )
    return out

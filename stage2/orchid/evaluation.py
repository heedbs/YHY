from __future__ import annotations
import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score

from . import CLASS_NAMES

PROBABILITY_COLUMNS = [f"p_{name}" for name in CLASS_NAMES]


def aggregate_patient_predictions(patches: pd.DataFrame) -> pd.DataFrame:
    if patches.empty:
        raise RuntimeError("empty prediction frame")
    if patches.groupby("patient_id")["target"].nunique().gt(1).any():
        raise RuntimeError("patient target contradiction")
    values = patches[PROBABILITY_COLUMNS].to_numpy(float)
    if not np.isfinite(values).all() or np.max(np.abs(values.sum(1) - 1.0)) > 1e-5:
        raise RuntimeError("invalid patch probabilities")
    images = patches.groupby(["patient_id", "source_path", "target"], as_index=False).agg(
        **{column: (column, "mean") for column in PROBABILITY_COLUMNS}, n_patches=("patch_uid", "count")
    )
    patients = images.groupby(["patient_id", "target"], as_index=False).agg(
        **{column: (column, "mean") for column in PROBABILITY_COLUMNS},
        n_images=("source_path", "count"), n_patches=("n_patches", "sum"),
    )
    probs = patients[PROBABILITY_COLUMNS].to_numpy(float)
    if not np.isfinite(probs).all() or np.max(np.abs(probs.sum(1) - 1.0)) > 1e-5:
        raise RuntimeError("invalid patient probabilities")
    patients["predicted_class"] = probs.argmax(1)
    return patients


def patient_metrics(patients: pd.DataFrame) -> dict:
    y = patients["target"].to_numpy(int)
    pred = patients["predicted_class"].to_numpy(int)
    prob = patients[PROBABILITY_COLUMNS].to_numpy(float)
    present = sorted(set(y.tolist()))
    auc = float("nan")
    if present == [0, 1, 2]:
        auc = float(roc_auc_score(np.eye(3)[y], prob, average="macro", multi_class="ovr"))
    return {
        "patient_n": int(len(y)), "macro_ovr_auroc": auc,
        "macro_ovr_auroc_defined": present == [0, 1, 2],
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "accuracy": float(accuracy_score(y, pred)),
        "brier_score": float(np.mean(np.sum((prob - np.eye(3)[y]) ** 2, axis=1))),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1, 2]).tolist(),
    }


def metric_key(metrics: dict) -> tuple:
    auc = metrics["macro_ovr_auroc"]
    return (metrics["macro_f1"], -float("inf") if np.isnan(auc) else auc, -metrics["loss"], -metrics["epoch"])

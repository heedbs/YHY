"""Exact pure functions extracted from final analysis; no analysis runs on import."""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
def safe_value(function):
    try:
        return float(function()), True
    except ValueError:
        return float("nan"), False

def classwise_metrics(target: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cm = confusion_matrix(target, pred, labels=[0, 1, 2])
    sensitivity, specificity, precision = [], [], []
    for index in range(3):
        tp = int(cm[index, index])
        fn = int(cm[index, :].sum() - tp)
        fp = int(cm[:, index].sum() - tp)
        tn = int(cm.sum() - tp - fn - fp)
        sensitivity.append(tp / (tp + fn) if tp + fn else float("nan"))
        specificity.append(tn / (tn + fp) if tn + fp else float("nan"))
        precision.append(tp / (tp + fp) if tp + fp else float("nan"))
    return cm, np.array(sensitivity), np.array(specificity), np.array(precision)

def calculate_metrics(frame: pd.DataFrame) -> dict:
    probability_columns = ["p_normal", "p_osmf", "p_oscc"]
    probabilities = frame[probability_columns].to_numpy(dtype=np.float64)
    target = frame["target"].to_numpy(dtype=np.int64)
    pred = frame["predicted_class"].to_numpy(dtype=np.int64)
    cm, sensitivity, specificity, precision = classwise_metrics(target, pred)
    one_hot = np.eye(3, dtype=np.float64)[target]
    auroc, auroc_defined = safe_value(lambda: roc_auc_score(target, probabilities, multi_class="ovr", average="macro", labels=[0, 1, 2]))
    auprc, auprc_defined = safe_value(lambda: average_precision_score(one_hot, probabilities, average="macro"))
    return {
        "patient_n": int(len(frame)),
        "accuracy": float(accuracy_score(target, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(target, pred)),
        "macro_f1": float(f1_score(target, pred, average="macro", labels=[0, 1, 2], zero_division=0)),
        "macro_ovr_auroc": auroc,
        "macro_ovr_auroc_defined": auroc_defined,
        "macro_ovr_auprc": auprc,
        "macro_ovr_auprc_defined": auprc_defined,
        "brier_score": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "normal_sensitivity": sensitivity[0], "osmf_sensitivity": sensitivity[1], "oscc_sensitivity": sensitivity[2],
        "normal_specificity": specificity[0], "osmf_specificity": specificity[1], "oscc_specificity": specificity[2],
        "normal_precision": precision[0], "osmf_precision": precision[1], "oscc_precision": precision[2],
        "confusion_matrix": json.dumps(cm.tolist(), separators=(",", ":")),
        "prediction_nan_count": int(np.isnan(probabilities).sum()),
        "prediction_inf_count": int(np.isinf(probabilities).sum()),
        "probability_row_sum_max_abs_error": float(np.abs(probabilities.sum(axis=1) - 1.0).max()),
        "argmax_mismatch_count": int((pred != probabilities.argmax(axis=1)).sum()),
        "single_class_prediction": bool(np.unique(pred).size == 1),
    }

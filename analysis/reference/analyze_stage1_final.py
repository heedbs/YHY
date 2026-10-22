#!/usr/bin/env python3
"""Read-only final patient-level audit and descriptive analysis for minimal v3 Stage 1."""
from __future__ import annotations
import os

import hashlib
import json
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)

ROOT = Path(str(Path(os.environ["ORCHID_PRIVATE_ROOT"]) / 'orchid_comparative_v3_minimal'))
OUTPUT = Path(str(Path(os.environ["ORCHID_PRIVATE_ROOT"]) / 'orchid_comparative_v3_stage1_final_analysis_20260830_175158'))
SPLIT_PATH = Path(str(Path(os.environ["ORCHID_PRIVATE_ROOT"]) / 'orchid_comparative_protocol_v2_freeze_20260824_120821/final_outer_inner_split_manifest.csv'))
SPEC_PATH = ROOT / "SCIENTIFIC_SPEC.md"
CLASS_NAMES = ["normal", "osmf", "oscc"]
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}
METRICS = ["accuracy", "balanced_accuracy", "macro_f1", "macro_ovr_auroc", "macro_ovr_auprc", "brier_score"]
ERROR_PATTERNS = {
    "traceback": re.compile(r"traceback", re.I),
    "cuda_oom": re.compile(r"cuda out of memory|outofmemoryerror", re.I),
    "nonfinite": re.compile(r"nonfinite|nan|inf", re.I),
    "fallback": re.compile(r"fallback", re.I),
    "assertion_failure": re.compile(r"assertionerror|assert .*failed", re.I),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def log_pattern_counts(output_path: Path) -> dict:
    counts = {key: 0 for key in ERROR_PATTERNS}
    for path in list(output_path.glob("*.log")) + list(output_path.glob("*.out")) + list(output_path.glob("*.err")):
        text = path.read_text(errors="replace")
        for key, pattern in ERROR_PATTERNS.items():
            counts[key] += len(pattern.findall(text))
    return counts


def expected_test_mapping(split: pd.DataFrame) -> dict[str, tuple[int, int]]:
    rows = split[split["split"] == "outer_test"]
    result = {}
    for _, row in rows.iterrows():
        result[str(row["patient_id"])] = (int(row["outer_fold"]), CLASS_TO_INDEX[str(row["class"])])
    return result


def summarize_distribution(values: pd.Series, metric: str) -> dict:
    numeric = pd.to_numeric(values, errors="coerce")
    return {
        f"{metric}_mean": float(numeric.mean()),
        f"{metric}_sd": float(numeric.std(ddof=1)),
        f"{metric}_median": float(numeric.median()),
        f"{metric}_min": float(numeric.min()),
        f"{metric}_max": float(numeric.max()),
    }


def write_figure_data(model: str, frame: pd.DataFrame, fold_metrics: pd.DataFrame) -> None:
    probabilities = frame[["p_normal", "p_osmf", "p_oscc"]].to_numpy(dtype=np.float64)
    target = frame["target"].to_numpy(dtype=np.int64)
    pred = frame["predicted_class"].to_numpy(dtype=np.int64)
    for class_index, class_name in enumerate(CLASS_NAMES):
        y_binary = (target == class_index).astype(int)
        fpr, tpr, thresholds = roc_curve(y_binary, probabilities[:, class_index])
        pd.DataFrame({"model": model, "class": class_name, "fpr": fpr, "tpr": tpr, "threshold": thresholds}).to_csv(OUTPUT / f"figure_pooled_roc_{model.lower().replace('+', '_').replace('-', '_')}_{class_name}.csv", index=False)
        precision, recall, thresholds = precision_recall_curve(y_binary, probabilities[:, class_index])
        threshold_full = np.concatenate([thresholds, [np.nan]])
        pd.DataFrame({"model": model, "class": class_name, "precision": precision, "recall": recall, "threshold": threshold_full}).to_csv(OUTPUT / f"figure_pooled_pr_{model.lower().replace('+', '_').replace('-', '_')}_{class_name}.csv", index=False)
    cm = confusion_matrix(target, pred, labels=[0, 1, 2])
    normalized = cm / cm.sum(axis=1, keepdims=True)
    pd.DataFrame(normalized, index=CLASS_NAMES, columns=CLASS_NAMES).rename_axis("true_class").reset_index().to_csv(OUTPUT / f"figure_normalized_confusion_{model.lower().replace('+', '_').replace('-', '_')}.csv", index=False)
    fold_metrics[fold_metrics["model"] == model].to_csv(OUTPUT / f"figure_fold_metric_distribution_{model.lower().replace('+', '_').replace('-', '_')}.csv", index=False)
    pd.DataFrame({"model": model, "patient_id": frame["patient_id"], "outer_fold": frame["outer_fold"], "target": target, "p_normal": probabilities[:, 0], "p_osmf": probabilities[:, 1], "p_oscc": probabilities[:, 2], "predicted_class": pred}).to_csv(OUTPUT / f"figure_patient_probability_distribution_{model.lower().replace('+', '_').replace('-', '_')}.csv", index=False)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    queue = pd.read_csv(ROOT / "stage1_queue.csv").sort_values("queue_order")
    split = pd.read_csv(SPLIT_PATH)
    expected = expected_test_mapping(split)
    integrity_rows, fold_rows, class_rows, oof_by_model = [], [], [], {}
    any_integrity_failure = False

    for entry in queue.itertuples(index=False):
        output_path = Path(entry.output_path)
        config_path = Path(entry.config_path)
        status_path, best_path, last_path = output_path / "status.json", output_path / "checkpoints/best.pt", output_path / "checkpoints/last.pt"
        metrics_path, prediction_path = output_path / "final_metrics.json", output_path / "outer_patient_predictions.csv"
        required = [status_path, best_path, last_path, metrics_path, prediction_path, config_path]
        missing = [str(path) for path in required if not path.exists()]
        record = {"run_id": entry.run_id, "model": entry.model_name, "fold": int(entry.outer_fold), "seed": int(entry.seed), "output_path": str(output_path), "missing_required_files": "|".join(missing), "integrity_pass": False}
        if missing:
            record.update({"status": "MISSING", "exit_code": np.nan, "failure_reasons": "required_file_missing"})
            integrity_rows.append(record); any_integrity_failure = True; continue
        status = json.loads(status_path.read_text())
        config = json.loads(config_path.read_text())
        final = json.loads(metrics_path.read_text())
        frame = pd.read_csv(prediction_path)
        checkpoint_errors, identity_errors = [], []
        try:
            best = torch.load(best_path, map_location="cpu", weights_only=False)
            last = torch.load(last_path, map_location="cpu", weights_only=False)
            for label, checkpoint in (("best", best), ("last", last)):
                for key in ("config_sha256", "split_sha256", "code_sha256", "pretrained_sha256", "class_order", "config", "model", "optimizer", "scheduler"):
                    if key not in checkpoint:
                        checkpoint_errors.append(f"{label}:{key}_missing")
                if checkpoint.get("config_sha256") != sha256(config_path): identity_errors.append(f"{label}:config_sha")
                if checkpoint.get("split_sha256") != config.get("split_sha256"): identity_errors.append(f"{label}:split_sha")
                if checkpoint.get("code_sha256") != config.get("code_sha256"): identity_errors.append(f"{label}:code_sha")
                if checkpoint.get("pretrained_sha256") != config.get("pretrained_sha256"): identity_errors.append(f"{label}:pretrained_sha")
                if checkpoint.get("class_order") != config.get("class_order"): identity_errors.append(f"{label}:class_order")
        except Exception as error:
            checkpoint_errors.append(f"checkpoint_decode:{type(error).__name__}:{error}")
        metric = calculate_metrics(frame)
        prediction_identity_errors = []
        if not (frame["run_id"] == entry.run_id).all(): prediction_identity_errors.append("prediction_run_id")
        if not (frame["model_name"] == entry.model_name).all(): prediction_identity_errors.append("prediction_model")
        if not (frame["outer_fold"] == int(entry.outer_fold)).all(): prediction_identity_errors.append("prediction_fold")
        if not (frame["seed"] == int(entry.seed)).all(): prediction_identity_errors.append("prediction_seed")
        expected_patients = {patient: target for patient, (fold, target) in expected.items() if fold == int(entry.outer_fold)}
        actual_patients = frame.set_index("patient_id")["target"].to_dict()
        missing_patients = sorted(set(expected_patients) - set(actual_patients))
        extra_patients = sorted(set(actual_patients) - set(expected_patients))
        wrong_targets = sum(actual_patients[p] != expected_patients[p] for p in set(actual_patients) & set(expected_patients))
        duplicate_patients = int(frame["patient_id"].duplicated().sum())
        outer_access = final.get("outer_access", {})
        log_counts = log_pattern_counts(output_path)
        failure_reasons = []
        if status.get("status") != "COMPLETED": failure_reasons.append("status")
        if status.get("exit_code") != 0: failure_reasons.append("exit_code")
        if checkpoint_errors: failure_reasons.append("checkpoint_decode_or_schema")
        if identity_errors: failure_reasons.append("checkpoint_identity")
        if prediction_identity_errors: failure_reasons.append("prediction_identity")
        if metric["prediction_nan_count"] or metric["prediction_inf_count"] or metric["probability_row_sum_max_abs_error"] > 1e-5 or metric["argmax_mismatch_count"]: failure_reasons.append("prediction_numeric")
        if missing_patients or extra_patients or wrong_targets or duplicate_patients: failure_reasons.append("split_target_identity")
        if outer_access.get("evaluation") != 1 or outer_access.get("prediction_rows") != len(frame): failure_reasons.append("outer_evaluation")
        if any(log_counts.values()): failure_reasons.append("log_error_pattern")
        record.update({
            "status": status.get("status"), "exit_code": status.get("exit_code"), "final_epoch": status.get("current_epoch"), "best_epoch": status.get("best_epoch"),
            "checkpoint_decode_errors": "|".join(checkpoint_errors), "checkpoint_identity_errors": "|".join(identity_errors), "prediction_identity_errors": "|".join(prediction_identity_errors),
            "expected_outer_patients": len(expected_patients), "observed_outer_patients": len(frame), "missing_patient_count": len(missing_patients), "extra_patient_count": len(extra_patients), "duplicate_patient_count": duplicate_patients, "wrong_target_count": wrong_targets,
            "outer_loader_creation": outer_access.get("loader_creation"), "outer_evaluation": outer_access.get("evaluation"), "outer_prediction_rows": outer_access.get("prediction_rows"),
            **log_counts, **{key: metric[key] for key in ("prediction_nan_count", "prediction_inf_count", "probability_row_sum_max_abs_error", "argmax_mismatch_count")},
            "failure_reasons": "|".join(failure_reasons), "integrity_pass": not failure_reasons,
        })
        integrity_rows.append(record)
        if failure_reasons:
            any_integrity_failure = True
            continue
        fold_record = {"run_id": entry.run_id, "model": entry.model_name, "fold": int(entry.outer_fold), "seed": int(entry.seed), "final_epoch": status.get("current_epoch"), "best_epoch": status.get("best_epoch"), **metric}
        fold_rows.append(fold_record)
        for index, name in enumerate(CLASS_NAMES):
            class_rows.append({"analysis_level": "fold", "model": entry.model_name, "fold": int(entry.outer_fold), "class": name, "sensitivity": metric[f"{name}_sensitivity"], "specificity": metric[f"{name}_specificity"], "precision": metric[f"{name}_precision"]})
        oof_by_model.setdefault(entry.model_name, []).append(frame)

    integrity = pd.DataFrame(integrity_rows).sort_values(["model", "fold"])
    integrity.to_csv(OUTPUT / "stage1_run_integrity.csv", index=False)
    if any_integrity_failure or len(fold_rows) != 20:
        report = "# Stage 1 Final Analysis Report\n\n## Final status\n\n`INTEGRITY_FAILURE_METRIC_COMPARISON_STOPPED`\n\nOne or more run-level integrity requirements failed. Per the requested fail-stop, no model-level pooled metrics, comparisons, or figure data were produced. See `stage1_run_integrity.csv`.\n"
        (OUTPUT / "STAGE1_FINAL_ANALYSIS_REPORT.md").write_text(report)
        return

    folds = pd.DataFrame(fold_rows).sort_values(["model", "fold"])
    folds.to_csv(OUTPUT / "stage1_fold_metrics.csv", index=False)
    summary_rows, pooled_rows, confusion_rows, oof_integrity_rows = [], [], [], []
    canonical_patient_order, canonical_target = None, None
    for model in queue["model_name"].drop_duplicates().tolist():
        model_folds = folds[folds["model"] == model].sort_values("fold")
        summary = {"model": model, "completed_folds": len(model_folds)}
        for metric_name in METRICS:
            summary.update(summarize_distribution(model_folds[metric_name], metric_name))
        for name in CLASS_NAMES:
            summary.update(summarize_distribution(model_folds[f"{name}_sensitivity"], f"{name}_sensitivity"))
        summary_rows.append(summary)
        oof = pd.concat(oof_by_model[model], ignore_index=True).sort_values("patient_id").reset_index(drop=True)
        oof.to_csv(OUTPUT / f"stage1_oof_predictions_{model.lower().replace('+', '_').replace('-', '_')}.csv", index=False)
        actual = oof.set_index("patient_id")[["outer_fold", "target"]]
        expected_frame = pd.DataFrame([(patient, fold, target) for patient, (fold, target) in expected.items()], columns=["patient_id", "outer_fold", "target"]).sort_values("patient_id").reset_index(drop=True)
        missing = sorted(set(expected_frame.patient_id) - set(oof.patient_id)); extra = sorted(set(oof.patient_id) - set(expected_frame.patient_id)); duplicates = int(oof.patient_id.duplicated().sum())
        joined = expected_frame.merge(oof[["patient_id", "outer_fold", "target"]], on="patient_id", how="inner", suffixes=("_expected", "_observed"))
        wrong_fold = int((joined.outer_fold_expected != joined.outer_fold_observed).sum())
        wrong_target = int((joined.target_expected != joined.target_observed).sum())
        patient_order = tuple(oof.patient_id); target_order = tuple(oof.target.astype(int))
        if canonical_patient_order is None:
            canonical_patient_order, canonical_target = patient_order, target_order
            cross_model_ids_targets = True
        else:
            cross_model_ids_targets = patient_order == canonical_patient_order and target_order == canonical_target
        oof_integrity_rows.append({"model": model, "oof_rows": len(oof), "oof_unique_patients": oof.patient_id.nunique(), "expected_patients": 149, "missing_patient_count": len(missing), "extra_patient_count": len(extra), "duplicate_patient_count": duplicates, "wrong_fold_count": wrong_fold, "wrong_target_count": wrong_target, "cross_model_patient_id_target_order_match": cross_model_ids_targets, "oof_integrity_pass": len(oof) == 149 and oof.patient_id.nunique() == 149 and not missing and not extra and duplicates == 0 and wrong_fold == 0 and wrong_target == 0 and cross_model_ids_targets})
        pooled = calculate_metrics(oof)
        pooled_rows.append({"model": model, **pooled})
        cm = np.array(json.loads(pooled["confusion_matrix"]))
        confusion_rows.append({"model": model, "class_order": "normal=0|osmf=1|oscc=2", "confusion_matrix": pooled["confusion_matrix"]})
        for index, name in enumerate(CLASS_NAMES):
            class_rows.append({"analysis_level": "pooled_oof", "model": model, "fold": "all", "class": name, "sensitivity": pooled[f"{name}_sensitivity"], "specificity": pooled[f"{name}_specificity"], "precision": pooled[f"{name}_precision"]})
        write_figure_data(model, oof, folds)

    cv_summary = pd.DataFrame(summary_rows)
    pooled_df = pd.DataFrame(pooled_rows)
    oof_integrity = pd.DataFrame(oof_integrity_rows)
    if not oof_integrity["oof_integrity_pass"].all():
        raise RuntimeError("OOF integrity failure after run-level validation; metrics must not be reported")
    cv_summary.to_csv(OUTPUT / "stage1_model_cv_summary.csv", index=False)
    pooled_df.to_csv(OUTPUT / "stage1_pooled_oof_metrics.csv", index=False)
    pd.DataFrame(confusion_rows).to_csv(OUTPUT / "stage1_pooled_confusion_matrices.csv", index=False)
    pd.DataFrame(class_rows).to_csv(OUTPUT / "stage1_classwise_metrics.csv", index=False)
    oof_integrity.to_csv(OUTPUT / "stage1_oof_integrity.csv", index=False)

    # Bootstrap inferential specification is intentionally not invented post-result.
    pd.DataFrame([{
        "status": "NOT_CALCULATED", "reason": "SCIENTIFIC_SPEC.md does not prespecify bootstrap unit, seed, repetitions, stratification, or percentile method", "required_next_step": "prospective statistical analysis addendum before inferential analysis", "models": "all four"
    }]).to_csv(OUTPUT / "stage1_bootstrap_ci.csv", index=False)
    paired_rows = []
    model_order = queue["model_name"].drop_duplicates().tolist()
    comparisons = [("InceptionV3", candidate) for candidate in model_order if candidate != "InceptionV3"] + [("EfficientNet-Lite2", "EfficientNet-Lite2+Mamba")]
    lookup = {model: pd.read_csv(OUTPUT / f"stage1_oof_predictions_{model.lower().replace('+', '_').replace('-', '_')}.csv").sort_values("patient_id").reset_index(drop=True) for model in model_order}
    for reference, candidate in comparisons:
        for metric_name in METRICS:
            observed = calculate_metrics(lookup[candidate])[metric_name] - calculate_metrics(lookup[reference])[metric_name]
            paired_rows.append({"reference_model": reference, "candidate_model": candidate, "metric": metric_name, "observed_pooled_difference_candidate_minus_reference": observed, "ci_status": "NOT_CALCULATED", "ci_reason": "bootstrap inference not prespecified", "zero_in_ci": "NOT_APPLICABLE", "interpretation": "exploratory descriptive only"})
    pd.DataFrame(paired_rows).to_csv(OUTPUT / "stage1_paired_differences.csv", index=False)
    pd.DataFrame([{
        "status": "SELECTION_RULE_INCOMPLETE", "evidence": "SCIENTIFIC_SPEC.md defines Stage 2 as selected-model stability analysis but does not define the two-model selection rule or tie handling", "stage2_action": "not executed", "selection_action": "no models selected"
    }]).to_csv(OUTPUT / "stage1_selection_rule_audit.csv", index=False)

    environment = [
        f"timestamp={datetime.now(timezone.utc).astimezone().isoformat()}", f"python={sys.executable}", f"python_version={sys.version}", f"platform={platform.platform()}", f"torch={torch.__version__}", f"sklearn={sklearn.__version__}", f"numpy={np.__version__}", f"pandas={pd.__version__}", f"cuda_available={torch.cuda.is_available()}", f"analysis_gpu_used=false", f"scientific_spec_sha256={sha256(SPEC_PATH)}", f"split_sha256={sha256(SPLIT_PATH)}"
    ]
    (OUTPUT / "environment.txt").write_text("\n".join(environment) + "\n")
    report = [
        "# Stage 1 Final Analysis Report", "", "## Final status", "", "`PASS_DESCRIPTIVE_ANALYSIS_COMPLETE`", "", "All 20 completed runs passed required run integrity and all four 149-patient pooled OOF tables passed exact patient/fold/target identity checks.", "", "## Analysis-plan limitations", "", "Bootstrap CI and inferential paired CIs were not calculated because the frozen scientific specification does not predefine their resampling rules. Observed paired pooled differences are labelled exploratory descriptive only. Stage 2 selection is `SELECTION_RULE_INCOMPLETE`; no model was selected and Stage 2 was not run.", "", "## Scope", "", "No training, inference, GPU use, input modification, threshold optimization, patient/fold exclusion, Stage 2 execution, or NDB work was performed.",
    ]
    (OUTPUT / "STAGE1_FINAL_ANALYSIS_REPORT.md").write_text("\n".join(report) + "\n")

    manifest_paths = sorted(path for path in OUTPUT.iterdir() if path.is_file() and path.name != "SHA256_MANIFEST.txt")
    (OUTPUT / "SHA256_MANIFEST.txt").write_text("".join(f"{sha256(path)}  {path}\n" for path in manifest_paths))


if __name__ == "__main__":
    main()

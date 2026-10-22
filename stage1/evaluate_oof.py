#!/usr/bin/env python3
from __future__ import annotations
import os

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from orchid.data import SPLIT_MANIFEST
from orchid.evaluation import PROBABILITY_COLUMNS, patient_metrics


def validate_and_evaluate(paths: list[Path], model_name: str) -> dict:
    frames = [pd.read_csv(path) for path in paths]
    oof = pd.concat(frames, ignore_index=True)
    split = pd.read_csv(SPLIT_MANIFEST)
    expected = split[split["split"] == "outer_test"][["outer_fold", "patient_id", "class"]].drop_duplicates()
    expected = expected.rename(columns={"class": "expected_class"})
    if len(expected) != 149 or expected["patient_id"].duplicated().any():
        raise RuntimeError("frozen OOF identity is not 149 unique patients")
    if len(oof) != 149 or oof["patient_id"].duplicated().any():
        raise RuntimeError("OOF missing/extra/duplicate patient")
    merged = expected.merge(oof, on="patient_id", how="outer", indicator=True)
    if not (merged["_merge"] == "both").all():
        raise RuntimeError("OOF patient identity mismatch")
    if "outer_fold" in oof and (merged["outer_fold_x"] != merged["outer_fold_y"]).any():
        raise RuntimeError("OOF wrong-fold mapping")
    class_map = {"normal": 0, "osmf": 1, "oscc": 2}
    if (merged["target"] != merged["expected_class"].map(class_map)).any():
        raise RuntimeError("OOF wrong target")
    probabilities = merged[PROBABILITY_COLUMNS].to_numpy(float)
    if not np.isfinite(probabilities).all() or np.max(np.abs(probabilities.sum(1) - 1.0)) > 1e-5:
        raise RuntimeError("OOF probability integrity failure")
    metrics = patient_metrics(merged)
    metrics["model_name"] = model_name
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--model", required=True); parser.add_argument("--output", required=True); parser.add_argument("predictions", nargs="+")
    args = parser.parse_args(); result = validate_and_evaluate([Path(path) for path in args.predictions], args.model)
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=True) + "\n")

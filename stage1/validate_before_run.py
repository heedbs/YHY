#!/usr/bin/env python3
from __future__ import annotations
import os

import argparse
import json
from pathlib import Path

import pandas as pd

from orchid import CLASS_ORDER
from orchid.data import SPLIT_MANIFEST, validate_analytic_manifests
from orchid.models import verify_pretrained
from train_one_run import code_sha256
from orchid.checkpoint import sha256_file

ROOT = Path(__file__).resolve().parent


def validate_all() -> dict:
    integrity = validate_analytic_manifests()
    configs = sorted((ROOT / "configs").glob("stage1_*.json"))
    if len(configs) != 20:
        raise RuntimeError(f"expected 20 Stage 1 configs, found {len(configs)}")
    rows = []
    for path in configs:
        config = json.loads(path.read_text())
        if config["code_sha256"] != code_sha256() or config["split_sha256"] != sha256_file(SPLIT_MANIFEST):
            raise RuntimeError(f"hash mismatch: {path}")
        _, pretrained = verify_pretrained(config["model_name"])
        if config["pretrained_sha256"] != pretrained or config["class_order"] != CLASS_ORDER:
            raise RuntimeError(f"identity mismatch: {path}")
        rows.append((config["model_name"], int(config["outer_fold"]), int(config["seed"]), config["run_id"]))
    if len(set(rows)) != 20:
        raise RuntimeError("duplicate Stage 1 config identity")
    queue = pd.read_csv(ROOT / "stage1_queue.csv")
    if len(queue) != 20 or set(queue["run_id"]) != {row[3] for row in rows}:
        raise RuntimeError("queue/config mismatch")
    return {**integrity, "configs": len(configs), "queue_rows": len(queue), "code_sha256": code_sha256()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--json", action="store_true"); args = parser.parse_args()
    result = validate_all()
    print(json.dumps(result, indent=2) if args.json else "PASS")

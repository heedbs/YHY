#!/usr/bin/env python3
from __future__ import annotations
import os

import argparse
import contextlib
import hashlib
import json
import os
import tempfile
import traceback
from datetime import datetime
from pathlib import Path

import torch

from orchid import CLASS_ORDER
from orchid.checkpoint import sha256_file
from orchid.data import SPLIT_MANIFEST, validate_analytic_manifests
from orchid.models import verify_pretrained
from orchid.trainer import Trainer

ROOT = Path(__file__).resolve().parent
CODE_FILES = [ROOT / "train_one_run.py", *sorted((ROOT / "orchid").glob("*.py"))]
RESERVED = {"run_id", "status", "phase", "pid", "start_time", "end_time", "exit_code", "current_epoch", "optimizer_steps", "best_epoch", "best_metric", "error", "metrics"}


def timestamp():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_json(payload: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    with os.fdopen(fd, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=True); handle.write("\n")
    os.replace(temporary, path)


def write_status(path: Path, metadata: dict, metrics: dict):
    if RESERVED.intersection(metrics):
        raise RuntimeError("reserved status key collision")
    payload = dict(metadata); payload["metrics"] = dict(metrics)
    if set(payload) != RESERVED:
        raise RuntimeError("status schema mismatch")
    atomic_json(payload, path)


def code_sha256():
    digest = hashlib.sha256()
    for path in sorted(CODE_FILES):
        digest.update(str(path.relative_to(ROOT)).encode()); digest.update(sha256_file(path).encode())
    return digest.hexdigest()


def validate_config(path: Path, config: dict):
    required = {"run_id", "model_name", "outer_fold", "seed", "batch_size", "gradient_accumulation_steps", "effective_batch_size", "epochs", "early_stopping_patience", "optimizer", "scheduler", "lr", "weight_decay", "gradient_clip_norm", "num_workers", "class_order", "precision", "amp", "sampler", "loss", "checkpoint_criterion", "augmentation", "split_sha256", "pretrained_sha256", "code_sha256", "smoke_train_batches", "smoke_val_batches"}
    if required - config.keys():
        raise RuntimeError(f"incomplete config: {sorted(required - config.keys())}")
    if config["model_name"] not in {"Composite-Mamba", "Composite-NoMamba"} or int(config["epochs"]) != 40 or int(config["early_stopping_patience"]) != 12:
        raise RuntimeError("Stage 2 scientific identity mismatch")
    if config["checkpoint_criterion"] != "inner_patient_macro_f1|macro_ovr_auroc|validation_loss|earlier_epoch":
        raise RuntimeError("checkpoint criterion mismatch")
    observed = {"config_sha256": sha256_file(path), "split_sha256": sha256_file(SPLIT_MANIFEST), "code_sha256": code_sha256()}
    if config["split_sha256"] != observed["split_sha256"] or config["code_sha256"] != observed["code_sha256"]:
        raise RuntimeError("config/code/split SHA mismatch")
    _, pretrained = verify_pretrained(config["model_name"])
    if config["pretrained_sha256"] != pretrained or config["class_order"] != CLASS_ORDER or config["precision"] != "fp32" or config["amp"] is not False:
        raise RuntimeError("pretrained or FP32 identity mismatch")
    observed["pretrained_sha256"] = pretrained
    return observed


def status_metadata(config, phase, status, start, **updates):
    payload = {"run_id": config["run_id"], "status": status, "phase": phase, "pid": os.getpid(), "start_time": start, "end_time": None, "exit_code": None, "current_epoch": -1, "optimizer_steps": 0, "best_epoch": -1, "best_metric": None, "error": None}
    payload.update(updates)
    return payload


def run(args, config, hashes, output, start):
    phase = "SMOKE" if args.smoke else "FULL"
    status_path = output / "status.json"
    write_status(status_path, status_metadata(config, phase, "RUNNING", start), {})
    trainer = Trainer(config, args.device, output, args.smoke, hashes)
    metrics_path = output / "metrics.jsonl"; final = {"phase": phase}
    if args.smoke:
        epoch = 0
        train = trainer.train_epoch(epoch, int(config["smoke_train_batches"]))
        inner_metrics, inner_predictions = trainer.evaluate_inner(int(config["smoke_val_batches"])); inner_metrics["epoch"] = epoch
        checkpoint = trainer.save_epoch_checkpoint(epoch, trainer.consider(inner_metrics, epoch))
        decoded = torch.load(output / "checkpoints/last.pt", map_location="cpu", weights_only=False)
        if decoded["next_epoch"] != 1:
            raise RuntimeError("checkpoint save/load validation failed")
        trainer.load_best(); trainer.write_predictions(inner_predictions, output / "inner_patient_predictions.csv")
        if trainer.outer_counters() != {"loader_creation": 0, "evaluation": 0, "prediction_rows": 0}:
            raise RuntimeError("outer-test access in smoke")
        final.update({"train": train, "inner": inner_metrics, "checkpoint": checkpoint, "checkpoint_decode": True, "outer_access": trainer.outer_counters()})
        metrics_path.write_text(json.dumps(final, allow_nan=True) + "\n")
    else:
        for epoch in range(int(config["epochs"])):
            train = trainer.train_epoch(epoch); inner_metrics, inner_predictions = trainer.evaluate_inner(); inner_metrics["epoch"] = epoch
            checkpoint = trainer.save_epoch_checkpoint(epoch, trainer.consider(inner_metrics, epoch))
            with metrics_path.open("a") as handle:
                handle.write(json.dumps({"epoch": epoch, "train": train, "inner": inner_metrics, "checkpoint": checkpoint}, allow_nan=True) + "\n")
            if trainer.patience_counter >= int(config["early_stopping_patience"]):
                break
        trainer.load_best(); _, inner_predictions = trainer.evaluate_inner(); trainer.write_predictions(inner_predictions, output / "inner_patient_predictions.csv")
        outer_metrics, outer_predictions = trainer.evaluate_outer_once()
        if trainer.outer_counters()["evaluation"] != 1 or trainer.outer_counters()["loader_creation"] != 1:
            raise RuntimeError("outer-test was not evaluated exactly once")
        trainer.write_predictions(outer_predictions, output / "outer_patient_predictions.csv")
        final.update({"outer": outer_metrics, "outer_access": trainer.outer_counters()})
    atomic_json(final, output / "final_metrics.json")
    write_status(status_path, status_metadata(config, phase, "COMPLETED", start, end_time=timestamp(), exit_code=0, current_epoch=int(final["train"]["epoch"] if args.smoke else epoch), optimizer_steps=trainer.global_optimizer_steps, best_epoch=trainer.best_epoch, best_metric=trainer.best_metric), {"outer_access": trainer.outer_counters()})


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); parser.add_argument("--device", required=True); parser.add_argument("--output", required=True); parser.add_argument("--smoke", action="store_true"); args = parser.parse_args()
    config_path, output, start = Path(args.config).resolve(), Path(args.output).resolve(), timestamp()
    if output.exists():
        raise RuntimeError(f"output directory collision: {output}")
    config = json.loads(config_path.read_text()); stdout = stderr = None
    try:
        validate_analytic_manifests(); hashes = validate_config(config_path, config)
        output.mkdir(parents=True, exist_ok=False); stdout = (output / "stdout.log").open("a", buffering=1); stderr = (output / "stderr.log").open("a", buffering=1)
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            run(args, config, hashes, output, start)
    except Exception as error:
        if output.exists():
            atomic_json(status_metadata(config if "config" in locals() else {"run_id": "unknown"}, "SMOKE" if args.smoke else "FULL", "FAILED", start, end_time=timestamp(), exit_code=1, error=repr(error)), output / "status.json")
        raise
    finally:
        if stdout: stdout.close()
        if stderr: stderr.close()


if __name__ == "__main__":
    main()

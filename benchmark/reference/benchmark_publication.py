#!/usr/bin/env python3
"""Single-GPU, read-only publication benchmark driver for frozen ORCHID models."""
from __future__ import annotations
import os

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import platform
import random
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = Path(str(Path(os.environ["ORCHID_PRIVATE_ROOT"]) / 'orchid_efficiency_rerun_infrastructure_20260908_155714'))
SOURCE_RUNNER = SOURCE_ROOT / "benchmark_efficiency_raw.py"
SOURCE_SPEC = SOURCE_ROOT / "benchmark_rerun_spec.json"
SOURCE_RUNNER_SHA = "c51eb77390614f4516052c5a859ad0ff2fc5f3b35b6471bd45ca635f117ec1c5"
SOURCE_SPEC_SHA = "2a7c9ae49467cafee8dc7fc6a35faca94951a78940c52a988bd05d11dd512eed"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run_text(args: list[str]) -> str:
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()


def gpu_snapshot() -> dict[str, str]:
    query = "index,uuid,name,driver_version,memory.used,memory.total,utilization.gpu,temperature.gpu,clocks.sm,clocks.mem,power.draw,pstate"
    try:
        line = run_text(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"])
        keys = query.split(",")
        values = [v.strip() for v in line.split(",")]
        return dict(zip(keys, values, strict=True))
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def compute_processes() -> list[dict[str, str]]:
    query = "gpu_uuid,pid,process_name,used_gpu_memory"
    try:
        raw = run_text(["nvidia-smi", f"--query-compute-apps={query}", "--format=csv,noheader,nounits"])
    except subprocess.CalledProcessError:
        return []
    if not raw:
        return []
    return [dict(zip(query.split(","), [v.strip() for v in line.split(",")], strict=True)) for line in raw.splitlines()]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"No rows for {path.name}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def quantile(values: list[float], q: float) -> float:
    return float(np.quantile(np.asarray(values), q, method="linear"))


def load_source_runner():
    spec = importlib.util.spec_from_file_location("frozen_efficiency_runner", SOURCE_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not import frozen benchmark runner")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def validate_static() -> dict:
    if sha256(SOURCE_RUNNER) != SOURCE_RUNNER_SHA:
        raise RuntimeError("BLOCKED_IDENTITY_OR_PATH_MISMATCH: frozen runner SHA mismatch")
    if sha256(SOURCE_SPEC) != SOURCE_SPEC_SHA:
        raise RuntimeError("BLOCKED_IDENTITY_OR_PATH_MISMATCH: frozen spec SHA mismatch")
    spec = json.loads(SOURCE_SPEC.read_text())
    if spec["input_shape"] != [3, 256, 256]:
        raise RuntimeError("BLOCKED_MEASUREMENT_VALIDITY: frozen input shape is not 256x256")
    if spec["warmup_iterations"] != 30 or spec["timed_iterations"] != 100:
        raise RuntimeError("BLOCKED_MEASUREMENT_VALIDITY: warm-up/timed iterations mismatch")
    if spec["precision"] != "fp32_tf32_disabled":
        raise RuntimeError("BLOCKED_MEASUREMENT_VALIDITY: frozen precision mismatch")
    return spec


def model_runtime_check(row: dict, frozen, device) -> dict:
    """One finite-output check outside all timing boundaries."""
    import torch

    config, checkpoint = frozen.validate(row)
    models, data = frozen.import_code(Path(row["code_root"]))
    _, pretrained_sha = models.verify_pretrained(row["model_name"])
    if pretrained_sha != config["pretrained_sha256"]:
        raise RuntimeError("runtime pretrained SHA mismatch")
    model = models.create_model(row["model_name"], pretrained=False).to(device).float().eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    image = frozen.preloaded_image(data)
    x = frozen.transform(image, models.preprocessing(row["model_name"]), 1).to(device)
    with torch.inference_mode():
        out = model(x)
    if tuple(out.shape) != (1, 3) or not bool(torch.isfinite(out).all()):
        raise RuntimeError(f"invalid output shape or nonfinite output: {tuple(out.shape)}")
    module_records = [f"{name}:{module.__class__.__module__}.{module.__class__.__name__}" for name, module in model.named_modules() if module.__class__.__module__.startswith("mamba_ssm")]
    if row["model_name"] == "Composite-Mamba" and not module_records:
        raise RuntimeError("Mamba runtime module missing; fallback forbidden")
    if row["model_name"] == "Composite-NoMamba" and module_records:
        raise RuntimeError("NoMamba runtime contains mamba_ssm module")
    evidence = {"model_name": row["model_name"], "output_shape": str(tuple(out.shape)), "output_finite": True,
                "mamba_runtime_modules": "|".join(module_records) if module_records else "NONE",
                "mamba_fallback": False, "pretrained_sha256": pretrained_sha}
    del model, x, out
    torch.cuda.empty_cache()
    return evidence


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gpu-id", default="0")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if not args.gpu_id.isdigit():
        raise RuntimeError("GPU ID must be a nonnegative integer")
    if args.output.exists():
        raise RuntimeError(f"BLOCKED_OUTPUT_COLLISION: {args.output}")
    spec = validate_static()
    if args.dry_run:
        print(json.dumps({"status": "DRY_RUN_PASS", "output": str(args.output), "gpu_id": args.gpu_id,
                          "models": [r["model_name"] for r in spec["models"]], "measurement_blocks": 3}, indent=2))
        return
    before_processes = compute_processes()
    if before_processes:
        raise RuntimeError(f"BLOCKED_GPU_BUSY: compute processes present before run: {before_processes}")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    import torch
    if not torch.cuda.is_available() or "RTX 4500 Ada" not in torch.cuda.get_device_name(0):
        raise RuntimeError("BLOCKED_IDENTITY_OR_PATH_MISMATCH: required GPU unavailable")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    frozen = load_source_runner()
    args.output.mkdir(parents=True)
    started = datetime.now().astimezone().isoformat(timespec="seconds")
    (args.output / "benchmark_status.json").write_text(json.dumps({"status": "RUNNING", "started": started}, indent=2) + "\n")
    try:
        device = torch.device("cuda:0")
        runtime = [model_runtime_check(row, frozen, device) for row in spec["models"]]
        cells = [(row, mode, batch) for row in spec["models"] for mode in ["model_only", "preprocess_plus_forward_excluding_file_io"] for batch in [1, int(row["operational_batch_size"])]]
        raw_rows, summaries, block_provenance = [], [], []
        own_pid = str(os.getpid())
        for block in range(1, 4):
            order = list(cells)
            random.Random(int(spec["order_seed"]) + block).shuffle(order)
            for position, (row, mode, batch) in enumerate(order, 1):
                pre = gpu_snapshot()
                competing = [x for x in compute_processes() if x.get("pid") != own_pid]
                if competing:
                    raise RuntimeError(f"BLOCKED_GPU_BUSY: competing process before block {block} cell {position}: {competing}")
                common, latencies = frozen.run_cell(row, device, spec, "model_only" if mode == "model_only" else "end_to_end", batch, position)
                post = gpu_snapshot()
                competing_after = [x for x in compute_processes() if x.get("pid") != own_pid]
                if competing_after:
                    raise RuntimeError(f"BLOCKED_MEASUREMENT_VALIDITY: competing process during cell: {competing_after}")
                if not all(np.isfinite(latencies)) or any(x <= 0 for x in latencies):
                    raise RuntimeError("BLOCKED_MEASUREMENT_VALIDITY: invalid raw latency")
                common.update({"measurement_block": block, "mode": mode, "timing_boundary": "GPU-resident tensor forward only" if mode == "model_only" else "host-memory frozen preprocessing plus H2D plus forward; file I/O excluded", "gpu_pre": json.dumps(pre, sort_keys=True), "gpu_post": json.dumps(post, sort_keys=True), "gpu_interference_detected": False})
                raw_rows.extend(dict(common, iteration=i + 1, latency_ms=lat) for i, lat in enumerate(latencies))
                summaries.append(dict(common, latency_median_ms=float(np.median(latencies)), latency_mean_ms=float(np.mean(latencies)), latency_sd_ms=float(np.std(latencies, ddof=1)), latency_iqr_ms=quantile(latencies, .75) - quantile(latencies, .25), latency_p5_ms=quantile(latencies, .05), latency_p95_ms=quantile(latencies, .95), throughput_images_per_second=batch * 1000.0 / float(np.median(latencies))))
                block_provenance.append({"measurement_block": block, "execution_position": position, "model_name": row["model_name"], "mode": mode, "batch_size": batch, "started_snapshot": json.dumps(pre, sort_keys=True), "ended_snapshot": json.dumps(post, sort_keys=True), "competing_processes": "NONE"})
        write_csv(args.output / "EFFICIENCY_RAW_MEASUREMENTS.csv", raw_rows)
        write_csv(args.output / "EFFICIENCY_CELL_BLOCK_SUMMARY.csv", summaries)
        write_csv(args.output / "EXECUTION_BLOCK_PROVENANCE.csv", block_provenance)
        write_csv(args.output / "MODEL_RUNTIME_EVIDENCE.csv", runtime)
        import cv2, timm
        import mamba_ssm
        env = {"started": started, "ended": datetime.now().astimezone().isoformat(timespec="seconds"), "python": sys.executable, "python_version": sys.version, "platform": platform.platform(), "gpu": torch.cuda.get_device_name(0), "gpu_uuid": gpu_snapshot().get("uuid", "UNAVAILABLE"), "driver": gpu_snapshot().get("driver_version", "UNAVAILABLE"), "torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "timm": timm.__version__, "mamba_ssm": getattr(mamba_ssm, "__version__", "unknown"), "mamba_ssm_path": str(Path(mamba_ssm.__file__).resolve()), "opencv": cv2.__version__, "precision": "FP32", "inference_mode": True, "autocast": False, "tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "tf32_cudnn": torch.backends.cudnn.allow_tf32, "input_shape": [3, 256, 256], "measurement_blocks": 3, "warmup_iterations": 30, "timed_iterations": 100}
        (args.output / "EFFICIENCY_ENVIRONMENT.json").write_text(json.dumps(env, indent=2) + "\n")
        (args.output / "benchmark_status.json").write_text(json.dumps({"status": "COMPLETED", "started": started, "ended": env["ended"], "cells": len(summaries), "raw_rows": len(raw_rows)}, indent=2) + "\n")
    except Exception:
        if args.output.exists():
            (args.output / "benchmark_status.json").write_text(json.dumps({"status": "FAILED", "traceback": traceback.format_exc()}, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()

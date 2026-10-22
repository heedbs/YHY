from __future__ import annotations
import os

import hashlib
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rng_state() -> dict:
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch_cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state: dict) -> None:
    random.setstate(state["python"]); np.random.set_state(state["numpy"]); torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def atomic_save(payload: dict, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    torch.save(payload, temporary); os.replace(temporary, path)
    return sha256_file(path)


def load_resume(path: Path, expected: dict) -> dict:
    if path.name != "last.pt":
        raise RuntimeError("resume is allowed only from last.pt")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("checkpoint_kind") != "epoch_boundary":
        raise RuntimeError("checkpoint is not an epoch boundary")
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise RuntimeError(f"resume identity mismatch: {key}")
    required = {"model", "optimizer", "scheduler", "next_epoch", "best_metric", "best_epoch", "patience_counter", "rng", "config_sha256", "split_sha256", "code_sha256", "pretrained_sha256", "model_identity"}
    if required - checkpoint.keys():
        raise RuntimeError(f"incomplete checkpoint: {sorted(required - checkpoint.keys())}")
    return checkpoint

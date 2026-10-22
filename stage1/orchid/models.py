from __future__ import annotations
import os

import hashlib
import inspect
from pathlib import Path

import pandas as pd
import timm
import torch

from . import CLASS_ORDER
from .data import BASE, PreprocessConfig

PROVENANCE = BASE / "selected_model_pretrained_provenance.csv"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def provenance_row(model_name: str):
    frame = pd.read_csv(PROVENANCE)
    rows = frame[frame["model_name"] == model_name]
    if len(rows) != 1:
        raise RuntimeError(f"pretrained provenance mismatch for {model_name}")
    return rows.iloc[0]


def verify_pretrained(model_name: str) -> tuple[Path, str]:
    row = provenance_row(model_name)
    path = Path(row["local_cache_path"])
    observed = sha256_file(path)
    if observed != row["weight_sha256"]:
        raise RuntimeError(f"pretrained SHA mismatch for {model_name}")
    return path, observed


def preprocessing(model_name: str) -> PreprocessConfig:
    row = provenance_row(model_name)
    parse = lambda text: tuple(float(value) for value in str(text).strip("[]").split(","))
    return PreprocessConfig(parse(row["official_mean"]), parse(row["official_std"]))


def create_model(model_name: str, pretrained: bool = True):
    verify_pretrained(model_name)
    if model_name == "EfficientNet-Lite2+Mamba":
        from . import mamba_impl
        if not mamba_impl.HAS_MAMBA:
            raise RuntimeError("mamba_ssm unavailable; fallback forbidden")
        model = mamba_impl.OrchidEffLite2_MSMambaDG(num_classes=3, pretrained=pretrained)
        mamba_blocks = [(name, module) for name, module in model.named_modules() if module.__class__.__module__.startswith("mamba_ssm")]
        if not mamba_blocks:
            raise RuntimeError("runtime model contains no mamba_ssm block")
        identity = mamba_identity(model, mamba_blocks)
    else:
        identifier = str(provenance_row(model_name)["exact_timm_identifier"])
        model = timm.create_model(identifier, pretrained=pretrained, num_classes=3)
        identity = {"model_name": model_name, "identifier": identifier, "num_classes": 3, "class_order": CLASS_ORDER}
    model._orchid_identity = identity
    output_features = getattr(model, "num_classes", 3)
    if model_name == "EfficientNet-Lite2+Mamba":
        output_features = model.head.fc.out_features
    if int(output_features) != 3:
        raise RuntimeError("model output dimension is not 3")
    return model


def mamba_identity(model, blocks):
    import mamba_ssm
    import selective_scan_cuda
    from mamba_ssm.modules.mamba_simple import Mamba
    local_source = Path(inspect.getfile(__import__("orchid.mamba_impl", fromlist=["x"]))).resolve()
    return {
        "model_name": "EfficientNet-Lite2+Mamba", "class_order": CLASS_ORDER, "num_classes": 3,
        "local_source": str(local_source), "local_source_sha256": sha256_file(local_source),
        "mamba_ssm_version": getattr(mamba_ssm, "__version__", "unknown"),
        "mamba_source": str(Path(inspect.getfile(Mamba)).resolve()),
        "mamba_source_sha256": sha256_file(Path(inspect.getfile(Mamba)).resolve()),
        "selective_scan_binary": str(Path(selective_scan_cuda.__file__).resolve()),
        "selective_scan_binary_sha256": sha256_file(Path(selective_scan_cuda.__file__).resolve()),
        "mamba_block_names": [name for name, _ in blocks],
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
    }


def model_identity(model) -> dict:
    identity = dict(getattr(model, "_orchid_identity", {}))
    if identity.get("num_classes") != 3 or identity.get("class_order") != CLASS_ORDER:
        raise RuntimeError("model identity mismatch")
    return identity

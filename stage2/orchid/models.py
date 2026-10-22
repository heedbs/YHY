from __future__ import annotations
import os

import inspect
from pathlib import Path

import pandas as pd
import torch.nn as nn

from . import CLASS_ORDER, mamba_impl
from .checkpoint import sha256_file
from .data import BASE, PreprocessConfig

PROVENANCE = BASE / "selected_model_pretrained_provenance.csv"
VARIANTS = {"Composite-Mamba", "Composite-NoMamba"}


def _backbone_row():
    rows = pd.read_csv(PROVENANCE).query("model_name == 'EfficientNet-Lite2+Mamba'")
    if len(rows) != 1:
        raise RuntimeError("EfficientNet-Lite2+Mamba pretrained provenance mismatch")
    return rows.iloc[0]


def verify_pretrained(model_name: str) -> tuple[Path, str]:
    if model_name not in VARIANTS:
        raise RuntimeError(f"unsupported Stage 2 model: {model_name}")
    row = _backbone_row(); path = Path(row["local_cache_path"]); observed = sha256_file(path)
    if observed != row["weight_sha256"]:
        raise RuntimeError("pretrained SHA mismatch")
    return path, observed


def preprocessing(model_name: str) -> PreprocessConfig:
    if model_name not in VARIANTS:
        raise RuntimeError(f"unsupported Stage 2 model: {model_name}")
    row = _backbone_row()
    parse = lambda text: tuple(float(value) for value in str(text).strip("[]").split(","))
    return PreprocessConfig(parse(row["official_mean"]), parse(row["official_std"]))


class CompositeNoMamba(mamba_impl.OrchidEffLite2_MSMambaDG):
    """Frozen Composite-Mamba architecture with only token Mamba replaced by identity."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.head.mamba = nn.Identity()


def _identity(model, variant: str) -> dict:
    source = Path(inspect.getfile(mamba_impl)).resolve()
    return {"model_name": variant, "class_order": CLASS_ORDER, "num_classes": 3,
            "local_source": str(source), "local_source_sha256": sha256_file(source),
            "parameter_count": int(sum(p.numel() for p in model.parameters())),
            "mamba_parameter_names": sorted(name for name, _ in model.named_parameters() if name.startswith("head.mamba."))}


def create_model(model_name: str, pretrained: bool = True):
    verify_pretrained(model_name)
    if not mamba_impl.HAS_MAMBA:
        raise RuntimeError("mamba_ssm unavailable; fallback forbidden")
    cls = mamba_impl.OrchidEffLite2_MSMambaDG if model_name == "Composite-Mamba" else CompositeNoMamba
    model = cls(num_classes=3, pretrained=pretrained)
    if model_name == "Composite-Mamba":
        if not any(module.__class__.__module__.startswith("mamba_ssm") for module in model.modules()):
            raise RuntimeError("runtime model contains no mamba_ssm block")
    elif not isinstance(model.head.mamba, nn.Identity) or any(name.startswith("head.mamba.") for name, _ in model.named_parameters()):
        raise RuntimeError("Composite-NoMamba is not an exact identity control")
    if int(model.head.fc.out_features) != 3:
        raise RuntimeError("model output dimension is not 3")
    model._orchid_identity = _identity(model, model_name)
    return model


def model_identity(model) -> dict:
    identity = dict(getattr(model, "_orchid_identity", {}))
    if identity.get("model_name") not in VARIANTS or identity.get("num_classes") != 3 or identity.get("class_order") != CLASS_ORDER:
        raise RuntimeError("model identity mismatch")
    return identity

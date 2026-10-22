from __future__ import annotations
import os

import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from . import CLASS_TO_INDEX

BASE = Path(os.environ["ORCHID_MANIFEST_DIR"])
PATCH_MANIFEST = BASE / "final_unique_patch_manifest.csv"
SPLIT_MANIFEST = BASE / "final_outer_inner_split_manifest.csv"
EXCLUSION_MANIFEST = BASE / "final_duplicate_exclusion_manifest.csv"


@dataclass(frozen=True)
class PreprocessConfig:
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    interpolation: str = "bicubic"
    input_size: int = 256


class OrchidPatchDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, preprocess: PreprocessConfig, training: bool):
        self.frame = frame.reset_index(drop=True)
        self.preprocess = preprocess
        self.training = training

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        image = cv2.imread(str(row["cache_path"]), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"missing or undecodable cache: {row['cache_path']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_CUBIC)
        if self.training:
            if random.random() < 0.5:
                image = np.ascontiguousarray(image[:, ::-1])
            if random.random() < 0.5:
                image = np.ascontiguousarray(image[::-1])
            if random.random() < 0.25:
                image = np.ascontiguousarray(np.rot90(image))
        array = image.astype(np.float32) / 255.0
        mean = np.asarray(self.preprocess.mean, np.float32).reshape(1, 1, 3)
        std = np.asarray(self.preprocess.std, np.float32).reshape(1, 1, 3)
        array = (array - mean) / std
        tensor = torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))
        meta = {
            "uid": str(row["uid"]),
            "patient_id": str(row["patient_key"]),
            "source_path": str(row["path"]),
            "source_sha256": str(row["source_sha256"]),
            "true_class": str(row["cls"]),
        }
        return tensor, torch.tensor(CLASS_TO_INDEX[meta["true_class"]], dtype=torch.long), meta


def validate_analytic_manifests() -> dict:
    patch = pd.read_csv(PATCH_MANIFEST)
    split = pd.read_csv(SPLIT_MANIFEST)
    if len(patch) != 232689 or patch["uid"].duplicated().any():
        raise RuntimeError("patch count or UID integrity failure")
    if set(patch["cls"]) != set(CLASS_TO_INDEX):
        raise RuntimeError("wrong class order/taxonomy")
    missing = [path for path in patch["cache_path"] if not Path(path).is_file()]
    if missing:
        raise RuntimeError(f"missing cache patches: {len(missing)}")
    patients = set(patch["patient_key"].astype(str))
    if len(patients) != 149:
        raise RuntimeError(f"analytic patient count mismatch: {len(patients)}")
    if patch["source_sha256"].nunique() != 10072:
        raise RuntimeError("unique source-image count mismatch")
    counts = patch[["patient_key", "cls"]].drop_duplicates()["cls"].value_counts().to_dict()
    if counts != {"oscc": 104, "osmf": 31, "normal": 14}:
        raise RuntimeError(f"patient class counts mismatch: {counts}")
    for fold_id in range(5):
        fold = split[split["outer_fold"] == fold_id]
        groups = {part: set(fold.loc[fold["split"] == part, "patient_id"].astype(str)) for part in ("train", "inner_val", "outer_test")}
        if groups["train"] & groups["inner_val"] or groups["train"] & groups["outer_test"] or groups["inner_val"] & groups["outer_test"]:
            raise RuntimeError(f"patient overlap in fold {fold_id}")
        subsets = {part: patch[patch["patient_key"].astype(str).isin(ids)] for part, ids in groups.items()}
        for left, right in (("train", "inner_val"), ("train", "outer_test"), ("inner_val", "outer_test")):
            if set(subsets[left]["uid"]) & set(subsets[right]["uid"]):
                raise RuntimeError(f"patch UID leakage in fold {fold_id}")
            if set(subsets[left]["source_sha256"]) & set(subsets[right]["source_sha256"]):
                raise RuntimeError(f"source SHA leakage in fold {fold_id}")
    return {"patients": 149, "patches": len(patch), "sources": patch["source_sha256"].nunique(), "folds": 5}


def load_fold_frames(outer_fold: int):
    patch = pd.read_csv(PATCH_MANIFEST)
    split = pd.read_csv(SPLIT_MANIFEST)
    fold = split[split["outer_fold"] == outer_fold]
    frames = {}
    for part in ("train", "inner_val", "outer_test"):
        ids = set(fold.loc[fold["split"] == part, "patient_id"].astype(str))
        frames[part] = patch[patch["patient_key"].astype(str).isin(ids)].copy().reset_index(drop=True)
        contradictions = frames[part].groupby("patient_key")["cls"].nunique()
        if contradictions.gt(1).any():
            raise RuntimeError(f"patient target contradiction in {part}")
    return frames


def make_loader(frame, preprocess, training, batch_size, num_workers, generator=None):
    sampler = None
    if training:
        counts = frame["patient_key"].astype(str).value_counts()
        weights = frame["patient_key"].astype(str).map(lambda key: 1.0 / counts[key]).to_numpy()
        sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), len(weights), replacement=True, generator=generator)
    return DataLoader(
        OrchidPatchDataset(frame, preprocess, training), batch_size=batch_size,
        sampler=sampler, shuffle=False, num_workers=num_workers, drop_last=False,
        pin_memory=torch.cuda.is_available(), persistent_workers=num_workers > 0,
        generator=generator,
    )

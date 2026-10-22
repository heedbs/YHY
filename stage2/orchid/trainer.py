from __future__ import annotations
import os

import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from . import CLASS_NAMES, CLASS_ORDER
from .checkpoint import atomic_save, sha256_file
from .data import SPLIT_MANIFEST, load_fold_frames, make_loader
from .evaluation import PROBABILITY_COLUMNS, aggregate_patient_predictions, metric_key, patient_metrics
from .models import create_model, model_identity, preprocessing, verify_pretrained


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Trainer:
    def __init__(self, config: dict, device: str, output: Path, smoke: bool, hashes: dict):
        self.config = config
        self.device = torch.device(device)
        self.output = output
        self.smoke = smoke
        self.hashes = hashes
        self.outer_loader_creation_count = 0
        self.outer_evaluation_count = 0
        self.outer_prediction_rows = 0
        if config["class_order"] != CLASS_ORDER or config["precision"] != "fp32" or config["amp"] is not False:
            raise RuntimeError("class order or FP32 identity mismatch")
        set_seed(int(config["seed"]))
        self.frames = load_fold_frames(int(config["outer_fold"]))
        self.preprocess = preprocessing(config["model_name"])
        self.inner_loader = make_loader(self.frames["inner_val"], self.preprocess, False, int(config["batch_size"]), int(config["num_workers"]))
        self.model = create_model(config["model_name"], pretrained=True).to(self.device)
        if any(parameter.dtype != torch.float32 for parameter in self.model.parameters() if parameter.is_floating_point()):
            raise RuntimeError("non-FP32 model parameter")
        self.identity = model_identity(self.model)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=float(config["lr"]), weight_decay=float(config["weight_decay"]))
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=int(config["epochs"]))
        self.start_epoch = 0
        self.best_metric = None
        self.best_epoch = -1
        self.patience_counter = 0
        self.global_optimizer_steps = 0
        self.runtime_evidence = {"mamba_input_shape": None, "mamba_output_shape": None, "nomamba_identity_exact": None}
        def capture_mamba(module, inputs, output):
            self.runtime_evidence["mamba_input_shape"] = list(inputs[0].shape)
            self.runtime_evidence["mamba_output_shape"] = list(output.shape)
            if self.config["model_name"] == "Composite-NoMamba":
                self.runtime_evidence["nomamba_identity_exact"] = bool(torch.equal(inputs[0], output))
        self._mamba_hook = self.model.head.mamba.register_forward_hook(capture_mamba)

    def _train_loader(self, epoch: int):
        generator = torch.Generator().manual_seed(int(self.config["seed"]) * 1_000_003 + epoch)
        return make_loader(self.frames["train"], self.preprocess, True, int(self.config["batch_size"]), int(self.config["num_workers"]), generator)

    def _first_parameter_snapshot(self):
        name, parameter = next((item for item in self.model.named_parameters() if item[1].requires_grad))
        return name, parameter.detach().cpu().clone()

    def train_epoch(self, epoch: int, max_batches: int | None = None) -> dict:
        self.model.train()
        loader = self._train_loader(epoch)
        accumulation = int(self.config["gradient_accumulation_steps"])
        expected_effective = int(self.config["batch_size"]) * accumulation
        if expected_effective != int(self.config["effective_batch_size"]):
            raise RuntimeError("effective batch mismatch")
        limit = len(loader) if max_batches is None else min(len(loader), int(max_batches))
        if limit <= 0:
            raise RuntimeError("zero train batches")
        expected_steps = math.ceil(limit / accumulation)
        tracked_name, parameter_before = self._first_parameter_snapshot()
        self.optimizer.zero_grad(set_to_none=True)
        samples_in_window = 0
        steps = 0
        loss_sum_total = 0.0
        samples_total = 0
        finite_losses = 0
        mamba_gradient_tensors = 0
        mamba_nonzero_gradient_tensors = 0
        for batch_index, (images, targets, _) in enumerate(loader):
            if batch_index >= limit:
                break
            images, targets = images.to(self.device), targets.to(self.device)
            logits = self.model(images)
            if logits.ndim != 2 or logits.shape[1] != 3 or not torch.isfinite(logits).all():
                raise RuntimeError("invalid training logits")
            loss_sum = F.cross_entropy(logits, targets, reduction="sum")
            if not torch.isfinite(loss_sum):
                raise RuntimeError("nonfinite loss")
            finite_losses += 1
            loss_sum.backward()
            batch_samples = int(targets.shape[0])
            samples_in_window += batch_samples
            loss_sum_total += float(loss_sum.detach().cpu())
            samples_total += batch_samples
            due = (batch_index + 1) % accumulation == 0 or batch_index + 1 == limit
            if due:
                nonzero = 0
                for parameter_name, parameter in self.model.named_parameters():
                    if parameter.grad is None:
                        continue
                    if not torch.isfinite(parameter.grad).all():
                        raise RuntimeError("nonfinite gradient")
                    parameter.grad.div_(float(samples_in_window))
                    nonzero += int(torch.count_nonzero(parameter.grad).item() > 0)
                    if parameter_name.startswith("head.mamba."):
                        mamba_gradient_tensors += 1
                        mamba_nonzero_gradient_tensors += int(torch.count_nonzero(parameter.grad).item() > 0)
                if nonzero == 0:
                    raise RuntimeError("all gradients are zero or None")
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.config["gradient_clip_norm"]))
                self.optimizer.step(); self.optimizer.zero_grad(set_to_none=True)
                steps += 1; self.global_optimizer_steps += 1; samples_in_window = 0
        if samples_in_window or steps != expected_steps or steps == 0:
            raise RuntimeError("gradient accumulation boundary failure")
        if self.config["model_name"] == "Composite-Mamba" and (mamba_gradient_tensors == 0 or mamba_nonzero_gradient_tensors == 0):
            raise RuntimeError("Mamba gradient evidence missing")
        if self.config["model_name"] == "Composite-NoMamba" and self.runtime_evidence["nomamba_identity_exact"] is not True:
            raise RuntimeError("NoMamba identity path evidence missing")
        self.scheduler.step()
        name, parameter_after = next((item for item in self.model.named_parameters() if item[0] == tracked_name))
        difference = float(torch.linalg.vector_norm(parameter_after.detach().cpu() - parameter_before))
        if difference <= 0 or not self.optimizer.state_dict()["state"]:
            raise RuntimeError("no parameter update or empty AdamW state")
        return {
            "epoch": epoch, "physical_batches": limit, "optimizer_steps": steps,
            "global_optimizer_steps": self.global_optimizer_steps,
            "finite_loss_count": finite_losses, "nonfinite_loss_count": 0,
            "parameter_name": name, "parameter_l2_difference": difference,
            "parameter_update": True, "adamw_state_entries": len(self.optimizer.state_dict()["state"]),
            "scheduler_steps": 1, "accumulation_steps": accumulation,
            "effective_batch_size": expected_effective, "samples": samples_total,
            "loss": loss_sum_total / samples_total, "mamba_gradient_tensors": mamba_gradient_tensors,
            "mamba_nonzero_gradient_tensors": mamba_nonzero_gradient_tensors,
            "mamba_input_shape": self.runtime_evidence["mamba_input_shape"],
            "mamba_output_shape": self.runtime_evidence["mamba_output_shape"],
            "nomamba_identity_exact": self.runtime_evidence["nomamba_identity_exact"],
        }

    @torch.no_grad()
    def _evaluate(self, loader, max_batches: int | None = None):
        self.model.eval()
        rows = []
        loss_sum = 0.0
        count = 0
        for batch_index, (images, targets, meta) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            images, targets = images.to(self.device), targets.to(self.device)
            logits = self.model(images)
            probabilities = torch.softmax(logits, 1)
            if not torch.isfinite(logits).all() or not torch.isfinite(probabilities).all():
                raise RuntimeError("nonfinite evaluation output")
            probability_array = probabilities.cpu().numpy()
            if np.max(np.abs(probability_array.sum(1) - 1.0)) > 1e-5:
                raise RuntimeError("probability row-sum mismatch")
            loss_sum += float(F.cross_entropy(logits, targets, reduction="sum").cpu())
            count += len(targets)
            for index in range(len(targets)):
                rows.append({
                    "patch_uid": meta["uid"][index], "patient_id": meta["patient_id"][index],
                    "source_path": meta["source_path"][index], "target": int(targets[index].cpu()),
                    **{column: float(probability_array[index, class_index]) for class_index, column in enumerate(PROBABILITY_COLUMNS)},
                })
        patches = pd.DataFrame(rows)
        patients = aggregate_patient_predictions(patches)
        metrics = patient_metrics(patients)
        metrics["loss"] = loss_sum / count
        metrics["patch_rows"] = len(patches)
        return metrics, patients

    def evaluate_inner(self, max_batches: int | None = None):
        return self._evaluate(self.inner_loader, max_batches)

    def _create_outer_loader(self):
        if self.smoke:
            raise RuntimeError("outer-test loader creation forbidden in smoke")
        self.outer_loader_creation_count += 1
        return make_loader(self.frames["outer_test"], self.preprocess, False, int(self.config["batch_size"]), int(self.config["num_workers"]))

    def evaluate_outer_once(self):
        if self.smoke:
            raise RuntimeError("outer-test evaluation forbidden in smoke")
        self.outer_evaluation_count += 1
        metrics, patients = self._evaluate(self._create_outer_loader())
        self.outer_prediction_rows += len(patients)
        return metrics, patients

    def outer_counters(self):
        return {"loader_creation": self.outer_loader_creation_count, "evaluation": self.outer_evaluation_count, "prediction_rows": self.outer_prediction_rows}

    def consider(self, metrics: dict, epoch: int) -> bool:
        candidate = dict(metrics); candidate["epoch"] = epoch
        improved = self.best_metric is None or metric_key(candidate) > metric_key(self.best_metric)
        if improved:
            self.best_metric = candidate; self.best_epoch = epoch; self.patience_counter = 0
        else:
            self.patience_counter += 1
        return improved

    def save_epoch_checkpoint(self, epoch: int, improved: bool) -> dict:
        payload = {
            "model": self.model.state_dict(), "epoch": epoch, "next_epoch": epoch + 1,
            "best_metric": self.best_metric, "best_epoch": self.best_epoch,
            "patience_counter": self.patience_counter, "early_stopping_patience": int(self.config["early_stopping_patience"]),
            "global_optimizer_steps": self.global_optimizer_steps,
            "config": self.config, "config_sha256": self.hashes["config_sha256"],
            "split_sha256": self.hashes["split_sha256"], "code_sha256": self.hashes["code_sha256"],
            "pretrained_sha256": self.hashes["pretrained_sha256"], "model_identity": self.identity,
            "checkpoint_kind": "epoch_boundary", "class_order": CLASS_ORDER,
        }
        last = atomic_save(payload, self.output / "checkpoints/last.pt")
        best = atomic_save(payload, self.output / "checkpoints/best.pt") if improved else None
        return {"last_sha256": last, "best_sha256": best}

    def load_best(self):
        checkpoint = torch.load(self.output / "checkpoints/best.pt", map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model"])

    def write_predictions(self, patients: pd.DataFrame, path: Path) -> None:
        frame = patients.copy()
        if frame["patient_id"].duplicated().any():
            raise RuntimeError("duplicate patient prediction")
        frame.insert(0, "outer_fold", int(self.config["outer_fold"]))
        frame.insert(0, "seed", int(self.config["seed"]))
        frame.insert(0, "model_name", self.config["model_name"])
        frame.insert(0, "run_id", self.config["run_id"])
        frame.to_csv(path, index=False)

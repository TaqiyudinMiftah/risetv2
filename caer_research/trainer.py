"""Reusable trainer for clean in-repository controlled experiments."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Callable, Mapping

import torch
from torch import nn

from .checkpointing import (
    load_checkpoint_payload,
    restore_rng_state,
    save_checkpoint,
    training_checkpoint,
    unwrap_model,
)
from .engine import evaluate, train_one_epoch


EpochCallback = Callable[[dict[str, Any]], None]


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: Any,
        val_loader: Any,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        output_dir: Path | str,
        config: Mapping[str, Any],
        scheduler: Any = None,
        epochs: int = 1,
        monitor: str = "macro_f1",
        patience: int = 10,
        use_amp: bool = False,
        grad_clip_max_norm: float | None = None,
        epoch_callback: EpochCallback | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = dict(config)
        self.scheduler = scheduler
        self.epochs = epochs
        self.monitor = monitor
        self.patience = patience
        self.use_amp = bool(use_amp and device.type == "cuda")
        self.grad_clip_max_norm = grad_clip_max_norm
        self.epoch_callback = epoch_callback
        self.logger = logger or logging.getLogger(__name__)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.best_path = self.output_dir / "best.pt"
        self.last_path = self.output_dir / "last.pt"
        self.history_json_path = self.output_dir / "history.json"
        self.history_csv_path = self.output_dir / "history.csv"
        self.start_epoch = 1
        self.best_metric = float("-inf")
        self.early_stopping_count = 0
        self.history: list[dict[str, Any]] = []

    def resume(self, checkpoint_path: Path | str) -> None:
        checkpoint = load_checkpoint_payload(checkpoint_path, map_location=self.device)
        unwrap_model(self.model).load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if checkpoint.get("scaler_state_dict") is not None:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        if checkpoint.get("rng_state") is None:
            raise ValueError("Clean training resume requires checkpoint RNG state.")
        restore_rng_state(checkpoint["rng_state"])
        self.start_epoch = int(checkpoint["epoch"]) + 1
        self.best_metric = float(checkpoint["best_metric"])
        self.early_stopping_count = int(checkpoint["early_stopping_count"])
        self.history = list(checkpoint.get("history", []))

    def _history_row(
        self,
        epoch: int,
        train_metrics: Mapping[str, Any],
        val_metrics: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "epoch": epoch,
            "train_loss": float(train_metrics["loss"]),
            "train_accuracy": float(train_metrics["accuracy"]),
            "val_loss": float(val_metrics["loss"]),
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_macro_f1": float(val_metrics["macro_f1"]),
            "val_weighted_f1": float(val_metrics["weighted_f1"]),
            "val_neutral_f1": float(val_metrics["per_class"]["Neutral"]["f1"]),
            "val_nll": float(val_metrics["nll"]),
            "val_ece_15": float(val_metrics["ece_15"]),
            "lr": float(self.optimizer.param_groups[0]["lr"]),
        }

    def _write_history(self) -> None:
        self.history_json_path.write_text(
            json.dumps(self.history, indent=2) + "\n",
            encoding="utf-8",
        )
        if self.history:
            with self.history_csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(self.history[0]), lineterminator="\n")
                writer.writeheader()
                writer.writerows(self.history)

    def _checkpoint(self, epoch: int) -> dict[str, Any]:
        return training_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            epoch=epoch,
            best_metric=self.best_metric,
            early_stopping_count=self.early_stopping_count,
            history=self.history,
            config=self.config,
        )

    def fit(self) -> list[dict[str, Any]]:
        self.model.to(self.device)
        for epoch in range(self.start_epoch, self.epochs + 1):
            train_metrics = train_one_epoch(
                self.model,
                self.train_loader,
                self.optimizer,
                self.criterion,
                self.device,
                scaler=self.scaler,
                use_amp=self.use_amp,
                grad_clip_max_norm=self.grad_clip_max_norm,
            )
            val_metrics = evaluate(
                self.model,
                self.val_loader,
                self.criterion,
                self.device,
                use_amp=self.use_amp,
            )
            row = self._history_row(epoch, train_metrics, val_metrics)
            monitored_value = float(val_metrics[self.monitor])
            improved = monitored_value > self.best_metric
            if improved:
                self.best_metric = monitored_value
                self.early_stopping_count = 0
            else:
                self.early_stopping_count += 1
            if self.scheduler is not None:
                self.scheduler.step()
            self.history.append(row)
            self._write_history()
            payload = self._checkpoint(epoch)
            save_checkpoint(payload, self.last_path)
            if improved:
                save_checkpoint(payload, self.best_path)
            if self.epoch_callback is not None:
                self.epoch_callback(row)
            self.logger.info(
                "epoch=%s train_loss=%.6f val_loss=%.6f %s=%.6f",
                epoch,
                row["train_loss"],
                row["val_loss"],
                self.monitor,
                monitored_value,
            )
            if self.early_stopping_count >= self.patience:
                break
        return self.history

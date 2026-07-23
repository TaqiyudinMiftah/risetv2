"""Reusable sample-weighted training and evaluation steps."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .metrics import classification_metrics


def extract_logits(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, dict) and isinstance(output.get("logits"), torch.Tensor):
        return output["logits"]
    raise TypeError("Model output must be logits tensor or a dict containing 'logits'.")


def move_batch(
    batch: dict[str, Any], device: torch.device
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
    """Move only the modality tensors present in a frozen dataset batch."""

    non_blocking = device.type == "cuda"
    face = batch.get("face")
    context = batch.get("context")
    if face is None and context is None:
        raise ValueError("Batch must contain at least one of 'face' or 'context'.")
    if face is not None and not isinstance(face, torch.Tensor):
        raise TypeError("Batch field 'face' must be a tensor when present.")
    if context is not None and not isinstance(context, torch.Tensor):
        raise TypeError("Batch field 'context' must be a tensor when present.")
    return (
        face.to(device, non_blocking=non_blocking) if face is not None else None,
        context.to(device, non_blocking=non_blocking) if context is not None else None,
        batch["label"].to(device, non_blocking=non_blocking),
    )


def train_one_epoch(
    model: nn.Module,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: Any = None,
    use_amp: bool = False,
    grad_clip_max_norm: float | None = None,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for batch in loader:
        face, context, labels = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = extract_logits(model(face, context))
            loss = criterion(logits, labels)
        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            if grad_clip_max_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip_max_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_max_norm)
            optimizer.step()
        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_correct += int(logits.argmax(dim=1).eq(labels).sum().item())
        total_samples += batch_size
    return {"loss": total_loss / total_samples, "accuracy": total_correct / total_samples}


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: Any,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = False,
) -> dict[str, Any]:
    model.eval()
    labels: list[int] = []
    predictions: list[int] = []
    confidences: list[float] = []
    image_paths: list[str] = []
    total_loss = 0.0
    total_samples = 0
    for batch in loader:
        face, context, target = move_batch(batch, device)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = extract_logits(model(face, context))
            loss = criterion(logits, target)
        probability = logits.softmax(dim=1)
        confidence, prediction = probability.max(dim=1)
        batch_size = target.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        labels.extend(target.cpu().tolist())
        predictions.extend(prediction.cpu().tolist())
        confidences.extend(confidence.cpu().tolist())
        image_paths.extend(batch.get("image_path", [""] * batch_size))
    metrics = classification_metrics(labels, predictions, confidences)
    metrics.update(
        {
            "loss": total_loss / total_samples,
            "nll": total_loss / total_samples,
            "samples": total_samples,
            "labels": labels,
            "predictions": predictions,
            "confidences": confidences,
            "image_paths": image_paths,
        }
    )
    return metrics

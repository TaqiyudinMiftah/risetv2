from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def accuracy_topk(
    logits: torch.Tensor,
    targets: torch.Tensor,
    topk: tuple[int, ...] = (1,),
) -> list[float]:
    """Compute top-k accuracies."""
    maxk = max(topk)
    batch_size = targets.size(0)
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    res: list[float] = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(float(correct_k.mul_(100.0 / batch_size).item()))
    return res


class MetricTracker:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.total_loss = 0.0
        self.total_samples = 0
        self.total_correct_top1 = 0
        self.total_correct_top5 = 0

    def update(self, loss: float, logits: torch.Tensor, labels: torch.Tensor) -> None:
        batch_size = labels.size(0)
        self.total_loss += loss * batch_size
        self.total_samples += batch_size
        acc1, acc5 = accuracy_topk(logits, labels, topk=(1, 5))
        self.total_correct_top1 += acc1 * batch_size / 100.0
        self.total_correct_top5 += acc5 * batch_size / 100.0

    @property
    def avg_loss(self) -> float:
        return self.total_loss / max(1, self.total_samples)

    @property
    def avg_acc1(self) -> float:
        return self.total_correct_top1 / max(1, self.total_samples) * 100.0

    @property
    def avg_acc5(self) -> float:
        return self.total_correct_top5 / max(1, self.total_samples) * 100.0

    def summary(self) -> dict[str, float]:
        return {
            "loss": self.avg_loss,
            "acc1": self.avg_acc1,
            "acc5": self.avg_acc5,
        }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader[Any],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    tracker = MetricTracker()

    pbar = tqdm(loader, desc="train", leave=False)
    for batch in pbar:
        face = batch["face_image"].to(device)
        context = batch["context_image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        out = model(face, context)
        loss = criterion(out["logits"], labels)
        loss.backward()
        optimizer.step()

        tracker.update(loss.item(), out["logits"].detach(), labels)
        pbar.set_postfix({"loss": f"{tracker.avg_loss:.4f}", "acc1": f"{tracker.avg_acc1:.2f}%"})

    return tracker.summary()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader[Any],
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    tracker = MetricTracker()

    for batch in tqdm(loader, desc="eval", leave=False):
        face = batch["face_image"].to(device)
        context = batch["context_image"].to(device)
        labels = batch["label"].to(device)

        out = model(face, context)
        loss = criterion(out["logits"], labels)
        tracker.update(loss.item(), out["logits"], labels)

    return tracker.summary()


@torch.no_grad()
def evaluate_per_class(
    model: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    index_to_label: dict[int, str],
) -> dict[str, Any]:
    model.eval()
    num_classes = len(index_to_label)
    correct_per_class = torch.zeros(num_classes, dtype=torch.long)
    total_per_class = torch.zeros(num_classes, dtype=torch.long)

    for batch in tqdm(loader, desc="eval_per_class", leave=False):
        face = batch["face_image"].to(device)
        context = batch["context_image"].to(device)
        labels = batch["label"].to(device)

        out = model(face, context)
        preds = out["logits"].argmax(dim=1)

        for c in range(num_classes):
            mask = labels == c
            if mask.any():
                total_per_class[c] += mask.sum().item()
                correct_per_class[c] += (preds[mask] == labels[mask]).sum().item()

    per_class_acc: dict[str, float] = {}
    for idx, label_name in index_to_label.items():
        total = int(total_per_class[idx])
        correct = int(correct_per_class[idx])
        per_class_acc[label_name] = (correct / total * 100.0) if total > 0 else 0.0

    overall_acc = float(correct_per_class.sum().item() / max(1, total_per_class.sum().item()) * 100.0)

    return {
        "overall_acc": overall_acc,
        "per_class_acc": per_class_acc,
    }

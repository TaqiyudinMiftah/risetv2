from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageDraw
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm


CLASS_NAMES = ["Anger", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]


def apply_face_mask(image: Image.Image, bbox: list[int], fill: tuple[int, int, int] = (0, 0, 0)) -> Image.Image:
    masked = image.copy()
    draw = ImageDraw.Draw(masked)
    x1, y1, x2, y2 = [int(v) for v in bbox]
    draw.rectangle([x1, y1, x2, y2], fill=fill)
    return masked


class CAERSTwoStreamDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        dataset_root: Path,
        split: str,
        face_transform: Any,
        context_transform: Any,
    ) -> None:
        self.dataset_root = dataset_root
        self.face_transform = face_transform
        self.context_transform = context_transform
        self.label_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}
        self.idx_to_label = {idx: name for name, idx in self.label_to_idx.items()}

        with manifest_path.open("r") as f:
            rows = [json.loads(line) for line in f]
        self.rows = [row for row in rows if row["split"] == split]
        if not self.rows:
            raise ValueError(f"No rows found for split={split} in {manifest_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        image = Image.open(self.dataset_root / row["image_path"]).convert("RGB")
        bbox = [int(v) for v in row["face_bbox"]]
        x1, y1, x2, y2 = bbox

        face = image.crop((x1, y1, x2, y2))
        context = apply_face_mask(image, bbox)
        label_name = row["label"]

        return {
            "face": self.face_transform(face),
            "context": self.context_transform(context),
            "label": torch.tensor(self.label_to_idx[label_name], dtype=torch.long),
            "label_name": label_name,
            "image_path": row["image_path"],
        }


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, pool: bool = True) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.pool = nn.MaxPool2d(2, 2) if pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(F.relu(self.bn(self.conv(x))))


class CAERCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = ConvBlock(3, 32)
        self.conv2 = ConvBlock(32, 64)
        self.conv3 = ConvBlock(64, 128)
        self.conv4 = ConvBlock(128, 256)
        self.conv5 = ConvBlock(256, 256, pool=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return self.conv5(x)


class ContextAttention(nn.Module):
    def __init__(self, channels: int = 256) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, 128, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 1, kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.conv2(F.relu(self.bn1(self.conv1(x))))
        b, _, h, w = a.shape
        a = F.softmax(a.view(b, -1), dim=1).view(b, 1, h, w)
        return x * a


class AdaptiveFusion(nn.Module):
    def __init__(self, channels: int = 256, num_classes: int = 7, dropout: float = 0.5) -> None:
        super().__init__()
        self.face_w1 = nn.Conv2d(channels, 128, kernel_size=1)
        self.face_w2 = nn.Conv2d(128, 1, kernel_size=1)
        self.ctx_w1 = nn.Conv2d(channels, 128, kernel_size=1)
        self.ctx_w2 = nn.Conv2d(128, 1, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Conv2d(channels * 2, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(128, num_classes, kernel_size=1),
        )

    def forward(self, face_feat: torch.Tensor, ctx_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if face_feat.shape[-2:] != ctx_feat.shape[-2:]:
            target_h = min(face_feat.shape[-2], ctx_feat.shape[-2])
            target_w = min(face_feat.shape[-1], ctx_feat.shape[-1])
            face_feat = F.adaptive_avg_pool2d(face_feat, (target_h, target_w))
            ctx_feat = F.adaptive_avg_pool2d(ctx_feat, (target_h, target_w))

        lam_f = F.adaptive_avg_pool2d(self.face_w2(F.relu(self.face_w1(face_feat))), 1).flatten(1)
        lam_c = F.adaptive_avg_pool2d(self.ctx_w2(F.relu(self.ctx_w1(ctx_feat))), 1).flatten(1)
        weights = F.softmax(torch.cat([lam_f, lam_c], dim=1), dim=1)
        fused = torch.cat([face_feat * weights[:, 0:1, None, None], ctx_feat * weights[:, 1:2, None, None]], dim=1)
        logits = F.adaptive_avg_pool2d(self.classifier(fused), 1).flatten(1)
        return logits, weights


class CAERNetS(nn.Module):
    def __init__(self, num_classes: int = 7, dropout: float = 0.5) -> None:
        super().__init__()
        self.face_encoder = CAERCNN()
        self.context_encoder = CAERCNN()
        self.context_attention = ContextAttention(256)
        self.fusion = AdaptiveFusion(256, num_classes, dropout)

    def forward(self, face: torch.Tensor, context: torch.Tensor) -> dict[str, torch.Tensor]:
        face_feat = self.face_encoder(face)
        ctx_feat = self.context_attention(self.context_encoder(context))
        logits, weights = self.fusion(face_feat, ctx_feat)
        return {"logits": logits, "fusion_weights": weights}


def default_checkpoint() -> Path:
    legacy = Path("checkpoints/best_caernet_s.pt")
    if legacy.exists():
        return legacy
    candidates = sorted(Path("checkpoints").glob("*/best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        return candidates[0]
    raise FileNotFoundError("No checkpoint found. Pass --checkpoint explicitly.")


def load_model(checkpoint_path: Path, device: torch.device) -> CAERNetS:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
    model = CAERNetS(num_classes=len(CLASS_NAMES)).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, use_amp: bool) -> dict[str, Any]:
    y_true: list[int] = []
    y_pred: list[int] = []
    image_paths: list[str] = []
    total_loss = 0.0
    total = 0
    criterion = nn.CrossEntropyLoss()

    for batch in tqdm(loader, desc="eval"):
        face = batch["face"].to(device, non_blocking=device.type == "cuda")
        context = batch["context"].to(device, non_blocking=device.type == "cuda")
        labels = batch["label"].to(device, non_blocking=device.type == "cuda")

        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(face, context)["logits"]
            loss = criterion(logits, labels)

        preds = logits.argmax(1)
        total_loss += loss.item() * labels.size(0)
        total += labels.size(0)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())
        image_paths.extend(batch["image_path"])

    return {"loss": total_loss / total, "labels": y_true, "preds": y_pred, "image_paths": image_paths}


def write_outputs(metrics: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = metrics["labels"]
    preds = metrics["preds"]

    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, labels=list(range(len(CLASS_NAMES))), zero_division=0
    )
    metrics_out = {
        "test_loss": float(metrics["loss"]),
        "test_acc": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
        "weighted_f1": float(f1_score(labels, preds, average="weighted")),
        "per_class": {
            CLASS_NAMES[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(len(CLASS_NAMES))
        },
    }

    (output_dir / "metrics.json").write_text(json.dumps(metrics_out, indent=2))
    with (output_dir / "test_predictions.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "label", "label_name", "pred", "pred_name", "correct"])
        writer.writeheader()
        for image_path, label, pred in zip(metrics["image_paths"], labels, preds):
            writer.writerow(
                {
                    "image_path": image_path,
                    "label": label,
                    "label_name": CLASS_NAMES[label],
                    "pred": pred,
                    "pred_name": CLASS_NAMES[pred],
                    "correct": label == pred,
                }
            )

    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False)
    plt.title("CAER-S CAER-Net Test Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=200)
    plt.close(fig)

    print(json.dumps(metrics_out, indent=2))
    print(classification_report(labels, preds, target_names=CLASS_NAMES, digits=4, zero_division=0))
    print(f"Saved evaluation outputs to: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CAER-Net-S checkpoint on CAER-S test data.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to best.pt or legacy best_caernet_s.pt.")
    parser.add_argument("--manifest", type=Path, default=Path("caers_manifest.jsonl"))
    parser.add_argument("--dataset-root", type=Path, default=Path("CAER-S"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true", help="Use CUDA autocast during evaluation.")
    parser.add_argument("--max-samples", type=int, default=0, help="Optional smoke-test limit for test samples.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint or default_checkpoint()
    output_dir = args.output_dir or checkpoint_path.parent / "eval_test"
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")

    face_transform = T.Compose([
        T.Resize((96, 96)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    context_transform = T.Compose([
        T.Resize((128, 171)),
        T.CenterCrop(112),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset: Dataset = CAERSTwoStreamDataset(
        manifest_path=args.manifest,
        dataset_root=args.dataset_root,
        split="test",
        face_transform=face_transform,
        context_transform=context_transform,
    )
    if args.max_samples > 0:
        dataset = Subset(dataset, range(min(args.max_samples, len(dataset))))

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = load_model(checkpoint_path, device)
    metrics = evaluate(model, loader, device, use_amp=bool(args.amp and device.type == "cuda"))
    write_outputs(metrics, output_dir)


if __name__ == "__main__":
    main()

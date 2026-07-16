"""Exact reusable form of the legacy notebook CAERNetS architecture."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, pool: bool = True) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(2, 2) if pool else nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.pool(F.relu(self.bn(self.conv(inputs))))


class NotebookEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = ConvBlock(3, 32)
        self.conv2 = ConvBlock(32, 64)
        self.conv3 = ConvBlock(64, 128)
        self.conv4 = ConvBlock(128, 256)
        self.conv5 = ConvBlock(256, 256, pool=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.conv1(inputs)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        return self.conv5(output)


class ContextAttention(nn.Module):
    def __init__(self, channels: int = 256) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, 128, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 1, kernel_size=3, padding=1, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        attention = self.conv2(F.relu(self.bn1(self.conv1(inputs))))
        batch, _, height, width = attention.shape
        attention = F.softmax(attention.view(batch, -1), dim=1).view(batch, 1, height, width)
        return inputs * attention


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

    def forward(
        self,
        face_features: torch.Tensor,
        context_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if face_features.shape[-2:] != context_features.shape[-2:]:
            target = (
                min(face_features.shape[-2], context_features.shape[-2]),
                min(face_features.shape[-1], context_features.shape[-1]),
            )
            face_features = F.adaptive_avg_pool2d(face_features, target)
            context_features = F.adaptive_avg_pool2d(context_features, target)

        face_weight = F.adaptive_avg_pool2d(
            self.face_w2(F.relu(self.face_w1(face_features))), 1
        ).flatten(1)
        context_weight = F.adaptive_avg_pool2d(
            self.ctx_w2(F.relu(self.ctx_w1(context_features))), 1
        ).flatten(1)
        weights = F.softmax(torch.cat([face_weight, context_weight], dim=1), dim=1)
        fused = torch.cat(
            [
                face_features * weights[:, 0:1, None, None],
                context_features * weights[:, 1:2, None, None],
            ],
            dim=1,
        )
        logits = F.adaptive_avg_pool2d(self.classifier(fused), 1).flatten(1)
        return logits, weights


class NotebookCAERNet(nn.Module):
    """Legacy notebook model; keep separate from the upstream/paper baseline."""

    def __init__(self, num_classes: int = 7, dropout: float = 0.5) -> None:
        super().__init__()
        self.face_encoder = NotebookEncoder()
        self.context_encoder = NotebookEncoder()
        self.context_attention = ContextAttention(256)
        self.fusion = AdaptiveFusion(256, num_classes, dropout)

    def forward(self, face: torch.Tensor, context: torch.Tensor) -> dict[str, torch.Tensor]:
        face_features = self.face_encoder(face)
        context_features = self.context_attention(self.context_encoder(context))
        logits, weights = self.fusion(face_features, context_features)
        return {"logits": logits, "fusion_weights": weights}

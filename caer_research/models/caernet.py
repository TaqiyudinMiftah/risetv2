"""Dependency-free port of the upstream-community CAER-Net architecture."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class Encoder(nn.Module):
    def __init__(
        self,
        num_kernels: list[int],
        kernel_size: int = 3,
        batch_norm: bool = True,
        max_pool: bool = True,
        maxpool_kernel_size: int = 2,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(num_kernels[index], num_kernels[index + 1], kernel_size, padding=padding)
                for index in range(len(num_kernels) - 1)
            ]
        )
        self.bn = (
            nn.ModuleList([nn.BatchNorm2d(channels) for channels in num_kernels[1:]])
            if batch_norm
            else None
        )
        self.max_pool = nn.MaxPool2d(maxpool_kernel_size) if max_pool else None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = inputs
        for index, convolution in enumerate(self.convs):
            output = convolution(output)
            if self.bn is not None:
                output = self.bn[index](output)
            output = F.relu(output)
            if self.max_pool is not None and index < len(self.convs) - 1:
                output = self.max_pool(output)
        return output


class TwoStreamNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        channels = [3, 32, 64, 128, 256, 256]
        self.face_encoding_module = Encoder(channels)
        self.context_encoding_module = Encoder(channels)
        self.attention_inference_module = Encoder([256, 128, 1], max_pool=False)

    def forward(self, face: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        face_features = self.face_encoding_module(face)
        context_features = self.context_encoding_module(context)
        attention = self.attention_inference_module(context_features)
        batch, channels, height, width = attention.shape
        attention = F.softmax(attention.view(batch, -1), dim=-1).view(
            batch, channels, height, width
        )
        return face_features, context_features * attention


class FusionNetwork(nn.Module):
    def __init__(
        self,
        use_face: bool = True,
        use_context: bool = True,
        concat: bool = False,
        num_classes: int = 7,
    ) -> None:
        super().__init__()
        self.face_bn = nn.BatchNorm1d(256)
        self.context_bn = nn.BatchNorm1d(256)
        self.use_face = use_face
        self.use_context = use_context
        self.concat = concat
        self.face_1 = nn.Linear(256, 128)
        self.face_2 = nn.Linear(128, 1)
        self.context_1 = nn.Linear(256, 128)
        self.context_2 = nn.Linear(128, 1)
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout()

    def forward(self, face: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        face = F.avg_pool2d(face, face.shape[2]).view(face.shape[0], -1)
        context = F.avg_pool2d(context, context.shape[2]).view(context.shape[0], -1)
        face, context = self.face_bn(face), self.context_bn(context)

        if not self.concat:
            face_weight = self.face_2(F.relu(self.face_1(face)))
            context_weight = self.context_2(F.relu(self.context_1(context)))
            weights = F.softmax(torch.cat([face_weight, context_weight], dim=-1), dim=-1)
            face = face * weights[:, 0].unsqueeze(dim=-1)
            context = context * weights[:, 1].unsqueeze(dim=-1)
        if not self.use_face:
            face = torch.zeros_like(face)
        if not self.use_context:
            context = torch.zeros_like(context)

        features = torch.cat([face, context], dim=-1)
        features = self.dropout(F.relu(self.fc1(features)))
        return self.fc2(features)


class CAERNet(nn.Module):
    """CAER-Net matching the upstream-community state-dict layout."""

    def __init__(self, use_face: bool = True, use_context: bool = True, concat: bool = False) -> None:
        super().__init__()
        self.two_stream_net = TwoStreamNetwork()
        self.fusion_net = FusionNetwork(use_face, use_context, concat)

    def forward(self, face: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        face_features, context_features = self.two_stream_net(face, context)
        return self.fusion_net(face_features, context_features)

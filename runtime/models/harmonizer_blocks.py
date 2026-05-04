from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        weight = self.weight.view(1, -1, 1, 1)
        bias = self.bias.view(1, -1, 1, 1)
        return (x - mean) * torch.rsqrt(var + self.eps) * weight + bias


class SimpleGate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, dim=1)
        return a * b


class NAFBlockLite(nn.Module):
    def __init__(self, channels: int, expansion: int = 2) -> None:
        super().__init__()
        hidden = channels * expansion
        self.norm1 = LayerNorm2d(channels)
        self.pw1 = nn.Conv2d(channels, hidden * 2, kernel_size=1)
        self.dw = nn.Conv2d(hidden * 2, hidden * 2, kernel_size=3, padding=1, groups=hidden * 2)
        self.gate = SimpleGate()
        self.pw2 = nn.Conv2d(hidden, channels, kernel_size=1)
        self.beta = nn.Parameter(torch.zeros(channels))
        self.norm2 = LayerNorm2d(channels)
        self.ffn1 = nn.Conv2d(channels, hidden * 2, kernel_size=1)
        self.ffn_gate = SimpleGate()
        self.ffn2 = nn.Conv2d(hidden, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pw1(self.norm1(x))
        y = self.dw(y)
        y = self.gate(y)
        y = self.pw2(y)
        x = x + y * self.beta.view(1, -1, 1, 1)
        y = self.ffn1(self.norm2(x))
        y = self.ffn_gate(y)
        y = self.ffn2(y)
        return x + y * self.gamma.view(1, -1, 1, 1)


class NAFEncoderLite(nn.Module):
    def __init__(
        self,
        in_channels: int = 9,
        channels: tuple[int, ...] = (32, 64, 128, 192),
        blocks: tuple[int, ...] = (2, 2, 4, 6),
    ) -> None:
        super().__init__()
        if len(channels) != len(blocks):
            raise ValueError("channels and blocks must have same length")
        self.stem = nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1)
        stages = []
        downs = []
        for index, (ch, n_blocks) in enumerate(zip(channels, blocks)):
            stages.append(nn.Sequential(*[NAFBlockLite(ch) for _ in range(n_blocks)]))
            if index < len(channels) - 1:
                downs.append(nn.Conv2d(ch, channels[index + 1], kernel_size=2, stride=2))
        self.stages = nn.ModuleList(stages)
        self.downs = nn.ModuleList(downs)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats = []
        x = self.stem(x)
        for index, stage in enumerate(self.stages):
            x = stage(x)
            feats.append(x)
            if index < len(self.downs):
                x = self.downs[index](x)
        return feats


def resize_inner(x: torch.Tensor, height: int, inner_width: int) -> torch.Tensor:
    return F.interpolate(x, size=(height, inner_width), mode="bilinear", align_corners=False)

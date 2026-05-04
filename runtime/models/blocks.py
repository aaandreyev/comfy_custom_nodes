from __future__ import annotations

import functools

import torch
import torch.nn.functional as F
from torch import nn


@functools.lru_cache(maxsize=8)
def _gaussian_kernel_1d(sigma: float, radius: int) -> torch.Tensor:
    xs = torch.arange(-radius, radius + 1, dtype=torch.float32)
    kernel = torch.exp(-(xs**2) / (2 * sigma * sigma))
    return kernel / kernel.sum()


class FiLMGenerator(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = max(channels // 2, 16)
        self.net = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.SiLU(),
            nn.Linear(hidden, channels * 2),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        stats = x.mean(dim=(-2, -1))
        gamma, beta = self.net(stats).chunk(2, dim=1)
        return gamma.unsqueeze(-1).unsqueeze(-1), beta.unsqueeze(-1).unsqueeze(-1)

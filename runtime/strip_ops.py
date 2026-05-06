from __future__ import annotations

import torch


def mask_bbox(mask: torch.Tensor) -> tuple[int, int, int, int]:
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError("mask_bbox expects mask shape [B,1,H,W]")
    support = mask.amax(dim=0)[0] > 0.5
    ys, xs = torch.where(support)
    if ys.numel() == 0:
        raise RuntimeError("empty mask")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1

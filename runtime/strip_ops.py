from __future__ import annotations

import torch


def mask_bbox(mask: torch.Tensor) -> tuple[int, int, int, int]:
    ys, xs = torch.where(mask[0, 0] > 0.5)
    if ys.numel() == 0:
        raise RuntimeError("empty mask")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1

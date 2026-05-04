from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F


Side = Literal["left", "right", "top", "bottom"]


@dataclass(frozen=True)
class StripSpec:
    strip_height: int = 1024
    outer_width: int = 128
    inner_width: int = 128


def canonicalize_strip(strip: torch.Tensor, side: Side) -> torch.Tensor:
    if side == "left":
        return strip
    if side == "right":
        return torch.flip(strip, dims=(-1,))
    if side == "top":
        return torch.rot90(strip, k=1, dims=(-2, -1))
    if side == "bottom":
        return torch.rot90(strip, k=3, dims=(-2, -1))
    raise ValueError(f"unsupported side: {side}")


def build_decay_mask(height: int, width: int, seam_x: int, inner_width: int) -> torch.Tensor:
    xs = torch.arange(width, dtype=torch.float32).view(1, 1, 1, width)
    t = ((xs - seam_x) / max(inner_width, 1)).clamp(0.0, 1.0)
    decay = 0.5 * (1.0 + torch.cos(math.pi * t))
    decay = torch.where(xs < seam_x, torch.zeros_like(decay), decay)
    return decay.expand(1, 1, height, width)


def _replicate_pad_height(strip: torch.Tensor, target_height: int) -> torch.Tensor:
    pad_h = max(target_height - strip.shape[-2], 0)
    if pad_h == 0:
        return strip
    return F.pad(strip, (0, 0, 0, pad_h), mode="replicate")


def _strip_origin_along_axis(bbox_start: int, image_extent: int, strip_len: int) -> int:
    if strip_len <= 0 or image_extent <= 0:
        return 0
    if strip_len >= image_extent:
        return 0
    max_start = image_extent - strip_len
    return int(min(max(0, bbox_start), max_start))


def extract_side_strip(
    image: torch.Tensor,
    bbox: tuple[int, int, int, int],
    side: Side,
    spec: StripSpec,
) -> tuple[torch.Tensor, dict]:
    if image.ndim != 3:
        raise ValueError("expected CHW image tensor")
    _, height, width = image.shape
    x0, y0, x1, y1 = bbox
    if side == "left":
        y_start = _strip_origin_along_axis(y0, height, spec.strip_height)
        outer = image[:, y_start : y_start + spec.strip_height, max(0, x0 - spec.outer_width) : x0]
        inner = image[:, y_start : y_start + spec.strip_height, x0 : min(width, x0 + spec.inner_width)]
        outer_pad = max(0, spec.outer_width - outer.shape[-1])
        inner_pad = max(0, spec.inner_width - inner.shape[-1])
        if outer_pad > 0:
            outer = image[:, y_start : y_start + spec.strip_height, x0 : x0 + 1].expand(-1, -1, spec.outer_width) if outer.shape[-1] == 0 else F.pad(outer, (outer_pad, 0), mode="replicate")
        if inner_pad > 0:
            inner = image[:, y_start : y_start + spec.strip_height, x0 - 1 : x0].expand(-1, -1, spec.inner_width) if inner.shape[-1] == 0 else F.pad(inner, (0, inner_pad), mode="replicate")
        strip = _replicate_pad_height(torch.cat([outer, inner], dim=-1), spec.strip_height)
        edge_padded = outer_pad + inner_pad
    elif side == "right":
        y_start = _strip_origin_along_axis(y0, height, spec.strip_height)
        inner = image[:, y_start : y_start + spec.strip_height, max(0, x1 - spec.inner_width) : x1]
        outer = image[:, y_start : y_start + spec.strip_height, x1 : min(width, x1 + spec.outer_width)]
        inner_pad = max(0, spec.inner_width - inner.shape[-1])
        outer_pad = max(0, spec.outer_width - outer.shape[-1])
        if inner_pad > 0:
            inner = image[:, y_start : y_start + spec.strip_height, x1 : x1 + 1].expand(-1, -1, spec.inner_width) if inner.shape[-1] == 0 else F.pad(inner, (inner_pad, 0), mode="replicate")
        if outer_pad > 0:
            outer = image[:, y_start : y_start + spec.strip_height, x1 - 1 : x1].expand(-1, -1, spec.outer_width) if outer.shape[-1] == 0 else F.pad(outer, (0, outer_pad), mode="replicate")
        strip = canonicalize_strip(_replicate_pad_height(torch.cat([inner, outer], dim=-1), spec.strip_height), "right")
        edge_padded = inner_pad + outer_pad
    elif side == "top":
        x_start = _strip_origin_along_axis(x0, width, spec.strip_height)
        outer = image[:, max(0, y0 - spec.outer_width) : y0, x_start : x_start + spec.strip_height]
        inner = image[:, y0 : min(height, y0 + spec.inner_width), x_start : x_start + spec.strip_height]
        outer_pad = max(0, spec.outer_width - outer.shape[-2])
        inner_pad = max(0, spec.inner_width - inner.shape[-2])
        if outer_pad > 0:
            outer = image[:, y0 : y0 + 1, x_start : x_start + spec.strip_height].expand(-1, spec.outer_width, -1) if outer.shape[-2] == 0 else F.pad(outer, (0, 0, outer_pad, 0), mode="replicate")
        if inner_pad > 0:
            inner = image[:, y0 - 1 : y0, x_start : x_start + spec.strip_height].expand(-1, spec.inner_width, -1) if inner.shape[-2] == 0 else F.pad(inner, (0, 0, 0, inner_pad), mode="replicate")
        strip = _replicate_pad_height(canonicalize_strip(torch.cat([outer, inner], dim=-2), "top"), spec.strip_height)
        edge_padded = outer_pad + inner_pad
    elif side == "bottom":
        x_start = _strip_origin_along_axis(x0, width, spec.strip_height)
        inner = image[:, max(0, y1 - spec.inner_width) : y1, x_start : x_start + spec.strip_height]
        outer = image[:, y1 : min(height, y1 + spec.outer_width), x_start : x_start + spec.strip_height]
        inner_pad = max(0, spec.inner_width - inner.shape[-2])
        outer_pad = max(0, spec.outer_width - outer.shape[-2])
        if inner_pad > 0:
            inner = image[:, y1 : y1 + 1, x_start : x_start + spec.strip_height].expand(-1, spec.inner_width, -1) if inner.shape[-2] == 0 else F.pad(inner, (0, 0, inner_pad, 0), mode="replicate")
        if outer_pad > 0:
            outer = image[:, y1 - 1 : y1, x_start : x_start + spec.strip_height].expand(-1, spec.outer_width, -1) if outer.shape[-2] == 0 else F.pad(outer, (0, 0, 0, outer_pad), mode="replicate")
        strip = _replicate_pad_height(canonicalize_strip(torch.cat([inner, outer], dim=-2), "bottom"), spec.strip_height)
        edge_padded = inner_pad + outer_pad
    else:
        raise ValueError(f"unsupported side: {side}")
    meta = {"edge_padded_pixels": int(edge_padded)}
    if side in {"left", "right"}:
        meta["y_start"] = int(y_start)
    else:
        meta["x_start"] = int(x_start)
    return strip, meta

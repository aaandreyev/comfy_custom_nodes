from __future__ import annotations

import math

import torch


def _hann_taper_from_t(t: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 - torch.cos(math.pi * t.clamp(0.0, 1.0)))


def _build_band_alpha(distance: torch.Tensor, band_width: float, fade_start: float) -> torch.Tensor:
    band_width = max(float(band_width), 1.0)
    min_fade_width = 2.0 if band_width > 2.0 else 1.0
    fade_start = min(max(float(fade_start), 0.0), max(band_width - min_fade_width, 0.0))
    fade_width = max(band_width - fade_start, min_fade_width)
    alpha = torch.where(
        distance <= fade_start,
        torch.ones_like(distance),
        1.0 - ((distance - fade_start) / fade_width).clamp(0.0, 1.0),
    )
    return _hann_taper_from_t(alpha)


def _edge_corner_taper(
    pos: torch.Tensor,
    start: int,
    end: int,
    extent: int,
    corner_px: float,
) -> torch.Tensor:
    top = torch.ones_like(pos)
    bottom = torch.ones_like(pos)
    if start > 0:
        top = _hann_taper_from_t(((pos - float(start)) / corner_px).clamp(0.0, 1.0))
    if end < extent:
        bottom = _hann_taper_from_t(((float(end) - pos) / corner_px).clamp(0.0, 1.0))
    return torch.minimum(top, bottom).clamp_min(1e-3)


def build_seam_local_weight_map(
    mask: torch.Tensor,
    bbox: tuple[int, int, int, int],
    side: str,
    inner_width: int,
    blend_falloff_px: int | None = None,
    power: float = 1.5,
) -> torch.Tensor:
    _, _, height, width = mask.shape
    x0, y0, x1, y1 = [int(x) for x in bbox]
    device, dtype = mask.device, mask.dtype
    yy = torch.arange(height, device=device, dtype=dtype).view(1, 1, height, 1)
    xx = torch.arange(width, device=device, dtype=dtype).view(1, 1, 1, width)
    bw = max(x1 - x0, 1)
    bh = max(y1 - y0, 1)
    if blend_falloff_px is None:
        fw = float(max(1, min(int(inner_width), bw)))
        fh = float(max(1, min(int(inner_width), bh)))
        fade_start_w = 0.0
        fade_start_h = 0.0
    else:
        fw = float(max(0, min(int(blend_falloff_px), bw)))
        fh = float(max(0, min(int(blend_falloff_px), bh)))
        band_w = float(max(1, min(int(inner_width), bw)))
        band_h = float(max(1, min(int(inner_width), bh)))
        fade_start_w = fw
        fade_start_h = fh
        fw = band_w
        fh = band_h
    cpx_h = float(max(12, min(32, int(inner_width) // 8)))
    cpx_w = float(max(12, min(32, int(inner_width) // 8)))
    if side == "left":
        d = (xx - float(x0)).clamp_min(0.0)
        base = _build_band_alpha(d, fw, fade_start_w)
        corner = _edge_corner_taper(yy, y0, y1, height, cpx_h)
    elif side == "right":
        d = (float(x1) - xx).clamp_min(0.0)
        base = _build_band_alpha(d, fw, fade_start_w)
        corner = _edge_corner_taper(yy, y0, y1, height, cpx_h)
    elif side == "top":
        d = (yy - float(y0)).clamp_min(0.0)
        base = _build_band_alpha(d, fh, fade_start_h)
        corner = _edge_corner_taper(xx, x0, x1, width, cpx_w)
    elif side == "bottom":
        d = (float(y1) - yy).clamp_min(0.0)
        base = _build_band_alpha(d, fh, fade_start_h)
        corner = _edge_corner_taper(xx, x0, x1, width, cpx_w)
    else:
        raise ValueError(f"unsupported side: {side}")
    if power != 1.0:
        base = base.clamp(0.0, 1.0).pow(power)
    return base * corner * mask


def merge_side_deltas(
    side_deltas: dict[str, torch.Tensor],
    mask: torch.Tensor,
    *,
    side_confidences: dict[str, torch.Tensor] | None = None,
    bbox: tuple[int, int, int, int] | None = None,
    inner_width: int | None = None,
    blend_falloff_px: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    del side_confidences
    if not side_deltas:
        zeros = torch.zeros(mask.shape[0], 3, mask.shape[-2], mask.shape[-1], device=mask.device, dtype=mask.dtype)
        return zeros, {}
    use_seam = bbox is not None and inner_width is not None and inner_width > 0
    weights: dict[str, torch.Tensor] = {}
    stack = []
    deltas = []
    for side, delta in side_deltas.items():
        support = (delta.abs().mean(dim=1, keepdim=True) > 1e-8).to(mask.dtype)
        weight = build_seam_local_weight_map(mask, bbox, side, int(inner_width), blend_falloff_px=blend_falloff_px) * support if use_seam else mask * support
        weights[side] = weight
        stack.append(weight)
        deltas.append(delta)
    w_stack = torch.stack(stack, dim=0)
    d_stack = torch.stack(deltas, dim=0)
    weighted_sum = (w_stack * d_stack).sum(dim=0)
    total_w = w_stack.sum(dim=0)
    coverage = total_w.clamp(0.0, 1.0)
    normalized = weighted_sum / total_w.clamp_min(1e-8)
    merged = torch.where(total_w > 1.0, normalized, weighted_sum)
    merged = torch.where(total_w > 1.0, normalized * coverage, merged)
    return merged * mask, weights

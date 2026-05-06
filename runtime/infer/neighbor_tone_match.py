from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from ..strip_ops import mask_bbox
from .merge_bands import merge_side_deltas


def _rgb_to_yuv(rgb: torch.Tensor) -> torch.Tensor:
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = -0.14713 * r - 0.28886 * g + 0.436 * b
    v = 0.615 * r - 0.51499 * g - 0.10001 * b
    return torch.cat([y, u, v], dim=1)


def _yuv_to_rgb(yuv: torch.Tensor) -> torch.Tensor:
    y, u, v = yuv[:, 0:1], yuv[:, 1:2], yuv[:, 2:3]
    r = y + 1.13983 * v
    g = y - 0.39465 * u - 0.58060 * v
    b = y + 2.03211 * u
    return torch.cat([r, g, b], dim=1)


def _lowpass_preserve_aspect(x: torch.Tensor, short_side: int) -> torch.Tensor:
    short_side = max(int(short_side), 1)
    height, width = x.shape[-2:]
    current_short = min(height, width)
    if current_short <= short_side:
        return x
    scale = float(short_side) / float(current_short)
    target_h = max(1, int(round(height * scale)))
    target_w = max(1, int(round(width * scale)))
    down = F.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=False)
    return F.interpolate(down, size=(height, width), mode="bilinear", align_corners=False)


def _resize_preserve_short_side(x: torch.Tensor, short_side: int) -> torch.Tensor:
    short_side = max(int(short_side), 1)
    height, width = x.shape[-2:]
    current_short = min(height, width)
    if current_short <= short_side:
        return x
    scale = float(short_side) / float(current_short)
    target_h = max(1, int(round(height * scale)))
    target_w = max(1, int(round(width * scale)))
    return F.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=False)


def _normalize_yuv_for_lookup(yuv: torch.Tensor) -> torch.Tensor:
    out = yuv.clone()
    out[:, 0:1] = out[:, 0:1].clamp(0.0, 1.0)
    out[:, 1:2] = ((out[:, 1:2] + 0.436) / 0.872).clamp(0.0, 1.0)
    out[:, 2:3] = ((out[:, 2:3] + 0.615) / 1.23).clamp(0.0, 1.0)
    return out


def _quantize_yuv(yuv: torch.Tensor, bins: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bins = max(int(bins), 2)
    norm = _normalize_yuv_for_lookup(yuv)
    q = (norm * float(bins - 1)).round().long().clamp(0, bins - 1)
    return q[:, 0], q[:, 1], q[:, 2]


def _build_delta_lookup(
    generated_strip: torch.Tensor,
    original_strip: torch.Tensor,
    *,
    bins: int = 24,
) -> dict[str, torch.Tensor]:
    if generated_strip.shape != original_strip.shape:
        raise ValueError("generated_strip and original_strip must have the same shape")
    device = generated_strip.device
    dtype = generated_strip.dtype
    batch = generated_strip.shape[0]
    delta = original_strip - generated_strip
    qy, qu, qv = _quantize_yuv(generated_strip, bins)
    delta_sum = torch.zeros(batch, bins, bins, bins, 3, device=device, dtype=dtype)
    counts = torch.zeros(batch, bins, bins, bins, device=device, dtype=dtype)
    global_delta = delta.mean(dim=(-2, -1))
    for b in range(batch):
        ys = qy[b].reshape(-1)
        us = qu[b].reshape(-1)
        vs = qv[b].reshape(-1)
        vals = delta[b].permute(1, 2, 0).reshape(-1, 3)
        flat_idx = ys * (bins * bins) + us * bins + vs
        delta_sum_flat = delta_sum[b].view(-1, 3)
        counts_flat = counts[b].view(-1)
        counts_flat.index_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=dtype))
        delta_sum_flat.index_add_(0, flat_idx, vals)
    return {
        "delta_sum": delta_sum,
        "counts": counts,
        "global_delta": global_delta,
        "bins": torch.tensor(int(bins), device=device),
    }


def _lookup_delta(
    generated_band: torch.Tensor,
    lookup: dict[str, torch.Tensor],
) -> torch.Tensor:
    bins = int(lookup["bins"].item())
    qy, qu, qv = _quantize_yuv(generated_band, bins)
    batch, _, height, width = generated_band.shape
    out = torch.zeros_like(generated_band)
    delta_sum = lookup["delta_sum"]
    counts = lookup["counts"]
    global_delta = lookup["global_delta"]
    for b in range(batch):
        side_sum = delta_sum[b][qy[b], qu[b], qv[b]]
        side_count = counts[b][qy[b], qu[b], qv[b]].unsqueeze(0)
        mean_delta = side_sum.permute(2, 0, 1) / side_count.clamp_min(1e-6)
        fallback = global_delta[b].view(3, 1, 1).expand(3, height, width)
        out[b] = torch.where(side_count > 0.0, mean_delta, fallback)
    return out


def _extract_outer_side_strip(
    image: torch.Tensor,
    bbox: tuple[int, int, int, int],
    side: str,
    band_width: int,
) -> torch.Tensor | None:
    x0, y0, x1, y1 = bbox
    _, _, height, width = image.shape
    if side == "left":
        band = min(int(band_width), x0)
        if band <= 0:
            return None
        return image[:, :, y0:y1, x0 - band : x0]
    if side == "right":
        band = min(int(band_width), width - x1)
        if band <= 0:
            return None
        return image[:, :, y0:y1, x1 : x1 + band]
    if side == "top":
        band = min(int(band_width), y0)
        if band <= 0:
            return None
        return image[:, :, y0 - band : y0, x0:x1]
    if side == "bottom":
        band = min(int(band_width), height - y1)
        if band <= 0:
            return None
        return image[:, :, y1 : y1 + band, x0:x1]
    raise ValueError(f"unsupported side: {side}")


def _canonicalize_side_strip(strip: torch.Tensor, side: str) -> torch.Tensor:
    if side == "left":
        return strip.flip(-1)
    if side == "right":
        return strip
    if side == "top":
        return strip.flip(-2)
    if side == "bottom":
        return strip
    raise ValueError(f"unsupported side: {side}")


def _extract_inner_side_band(
    image: torch.Tensor,
    bbox: tuple[int, int, int, int],
    side: str,
    band_width: int,
) -> torch.Tensor | None:
    x0, y0, x1, y1 = bbox
    _, _, height, width = image.shape
    if side == "left":
        band = min(int(band_width), x1 - x0)
        if band <= 0:
            return None
        return image[:, :, y0:y1, x0 : x0 + band]
    if side == "right":
        band = min(int(band_width), x1 - x0)
        if band <= 0:
            return None
        return image[:, :, y0:y1, x1 - band : x1]
    if side == "top":
        band = min(int(band_width), y1 - y0)
        if band <= 0:
            return None
        return image[:, :, y0 : y0 + band, x0:x1]
    if side == "bottom":
        band = min(int(band_width), y1 - y0)
        if band <= 0:
            return None
        return image[:, :, y1 - band : y1, x0:x1]
    raise ValueError(f"unsupported side: {side}")


def _place_side_delta(
    delta: torch.Tensor,
    bbox: tuple[int, int, int, int],
    side: str,
    full_shape: tuple[int, int, int, int],
) -> torch.Tensor:
    batch, channels, height, width = full_shape
    target = torch.zeros(batch, channels, height, width, device=delta.device, dtype=delta.dtype)
    x0, y0, x1, y1 = bbox
    if side == "left":
        target[:, :, y0:y1, x0 : x0 + delta.shape[-1]] = delta
    elif side == "right":
        target[:, :, y0:y1, x1 - delta.shape[-1] : x1] = delta.flip(-1)
    elif side == "top":
        target[:, :, y0 : y0 + delta.shape[-2], x0:x1] = delta
    elif side == "bottom":
        target[:, :, y1 - delta.shape[-2] : y1, x0:x1] = delta.flip(-2)
    else:
        raise ValueError(f"unsupported side: {side}")
    return target


def apply_neighbor_tone_match(
    reference_rgb: torch.Tensor,
    drift_source_rgb: torch.Tensor,
    generated_rgb: torch.Tensor,
    mask: torch.Tensor,
    *,
    inner_width: int,
    inner_falloff_px: int,
    process_left: bool,
    process_right: bool,
    process_top: bool,
    process_bottom: bool,
    downsample_short_side: int,
    luma_strength: float,
    chroma_strength: float,
) -> tuple[torch.Tensor, dict]:
    if reference_rgb.ndim != 4 or drift_source_rgb.ndim != 4 or generated_rgb.ndim != 4:
        raise ValueError("reference_rgb, drift_source_rgb, and generated_rgb must be BCHW")
    if reference_rgb.shape != generated_rgb.shape or reference_rgb.shape != drift_source_rgb.shape:
        raise ValueError("reference_rgb, drift_source_rgb, and generated_rgb must have the same shape")

    mask = mask.float()
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    elif mask.ndim != 4:
        raise ValueError("mask must be [B,H,W] or [B,1,H,W]")
    if mask.shape[-2:] != reference_rgb.shape[-2:]:
        mask = F.interpolate(mask, size=reference_rgb.shape[-2:], mode="nearest")
    mask = (mask > 0.5).float()

    bbox = mask_bbox(mask)
    x0, y0, x1, y1 = bbox
    sides = []
    if process_left and x0 > 0:
        sides.append("left")
    if process_right and x1 < reference_rgb.shape[-1]:
        sides.append("right")
    if process_top and y0 > 0:
        sides.append("top")
    if process_bottom and y1 < reference_rgb.shape[-2]:
        sides.append("bottom")
    if not sides:
        return generated_rgb, {"reason": "no_processable_sides", "side_deltas": {}, "weights": {}, "bbox": bbox}

    reference_yuv = _rgb_to_yuv(reference_rgb)
    drift_source_yuv = _rgb_to_yuv(drift_source_rgb)
    generated_yuv = _rgb_to_yuv(generated_rgb)
    channel_scale = reference_yuv.new_tensor([float(luma_strength), float(chroma_strength), float(chroma_strength)]).view(1, 3, 1, 1)

    side_deltas: dict[str, torch.Tensor] = {}
    per_side_meta: dict[str, dict] = {}
    for side in sides:
        reference_outer = _extract_outer_side_strip(reference_yuv, bbox, side, int(inner_width))
        drift_outer = _extract_outer_side_strip(drift_source_yuv, bbox, side, int(inner_width))
        generated_inner = _extract_inner_side_band(generated_yuv, bbox, side, int(inner_width))
        if reference_outer is None or drift_outer is None or generated_inner is None:
            continue
        reference_outer = _canonicalize_side_strip(reference_outer, side)
        drift_outer = _canonicalize_side_strip(drift_outer, side)
        generated_inner = _canonicalize_side_strip(generated_inner, side)
        reference_small = _resize_preserve_short_side(reference_outer, int(downsample_short_side))
        drift_small = _resize_preserve_short_side(drift_outer, int(downsample_short_side))
        lookup = _build_delta_lookup(drift_small, reference_small)
        delta_band = _lookup_delta(generated_inner, lookup) * channel_scale
        placed = _place_side_delta(delta_band, bbox, side, generated_yuv.shape)
        side_deltas[side] = placed
        per_side_meta[side] = {
            "band_shape": [int(v) for v in generated_inner.shape[-2:]],
            "lookup_strip_shape": [int(v) for v in drift_small.shape[-2:]],
            "mean_abs_delta": float(delta_band.abs().mean().item()),
        }

    if not side_deltas:
        return generated_rgb, {"reason": "no_valid_side_deltas", "side_deltas": {}, "weights": {}, "bbox": bbox}

    merged_delta, weights = merge_side_deltas(
        side_deltas,
        mask,
        bbox=bbox,
        inner_width=int(inner_width),
        blend_falloff_px=int(inner_falloff_px),
    )
    corrected_yuv = generated_yuv + merged_delta
    corrected_rgb = _yuv_to_rgb(corrected_yuv).clamp(0.0, 1.0)
    return corrected_rgb, {
        "reason": "applied",
        "bbox": bbox,
        "side_deltas": side_deltas,
        "weights": weights,
        "merged_delta": merged_delta,
        "per_side": per_side_meta,
    }


def write_neighbor_tone_debug(
    reference_rgb: torch.Tensor,
    generated_rgb: torch.Tensor,
    corrected_rgb: torch.Tensor,
    debug: dict,
    *,
    root: Path | None = None,
) -> Path:
    root = root or (Path.cwd() / "outputs" / "debug_previews" / datetime.now().strftime("%Y%m%d_%H%M%S"))
    root.mkdir(parents=True, exist_ok=True)
    _save_tensor(reference_rgb[0], root / "reference.png")
    _save_tensor(generated_rgb[0], root / "generated.png")
    _save_tensor(corrected_rgb[0], root / "corrected.png")
    diff = (corrected_rgb - generated_rgb).abs()
    _save_tensor((diff[0] / diff[0].amax().clamp_min(1e-6)), root / "correction_abs_diff.png")
    merged = debug.get("merged_delta")
    if isinstance(merged, torch.Tensor):
        _save_tensor(_signed_preview(merged[0]), root / "merged_delta.png")
    for side, delta in debug.get("side_deltas", {}).items():
        _save_tensor(_signed_preview(delta[0]), root / f"side_{side}_delta.png")
    for side, weight in debug.get("weights", {}).items():
        _save_tensor(weight[0].repeat(3, 1, 1), root / f"weight_map_{side}.png")
    summary = {
        "reason": debug.get("reason", "applied"),
        "bbox": list(debug.get("bbox", [])),
        "per_side": debug.get("per_side", {}),
        "debug_root": str(root),
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return root


def _signed_preview(x: torch.Tensor) -> torch.Tensor:
    out = x[:3].detach().clone()
    max_abs = out.abs().amax().clamp_min(1e-6)
    out = out / (2.0 * max_abs) + 0.5
    return out.clamp(0.0, 1.0)


def _save_tensor(x: torch.Tensor, path: Path) -> None:
    arr = (x.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).astype("uint8")
    Image.fromarray(arr).save(path)

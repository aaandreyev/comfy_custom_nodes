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


def _reduce_side_delta(delta: torch.Tensor, side: str) -> torch.Tensor:
    if side in {"left", "right"}:
        return delta.mean(dim=-1, keepdim=True)
    if side in {"top", "bottom"}:
        return delta.mean(dim=-2, keepdim=True)
    raise ValueError(f"unsupported side: {side}")


def _lowpass_side_profile(
    profile: torch.Tensor,
    *,
    side: str,
    original_strip_shape: tuple[int, int],
    short_side: int,
) -> torch.Tensor:
    short_side = max(int(short_side), 1)
    strip_h, strip_w = original_strip_shape
    strip_short = min(strip_h, strip_w)
    if strip_short <= short_side:
        return profile
    scale = float(short_side) / float(strip_short)
    if side in {"left", "right"}:
        length = profile.shape[-2]
        target_length = max(1, int(round(length * scale)))
        if target_length >= length:
            return profile
        down = F.interpolate(profile, size=(target_length, 1), mode="bilinear", align_corners=False)
        return F.interpolate(down, size=(length, 1), mode="bilinear", align_corners=False)
    length = profile.shape[-1]
    target_length = max(1, int(round(length * scale)))
    if target_length >= length:
        return profile
    down = F.interpolate(profile, size=(1, target_length), mode="bilinear", align_corners=False)
    return F.interpolate(down, size=(1, length), mode="bilinear", align_corners=False)


def _extract_side_pair(
    reference: torch.Tensor,
    generated: torch.Tensor,
    bbox: tuple[int, int, int, int],
    side: str,
    band_width: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    x0, y0, x1, y1 = bbox
    _, _, height, width = reference.shape
    if side == "left":
        band = min(int(band_width), x0, x1 - x0)
        if band <= 0:
            return None, None
        outer = reference[:, :, y0:y1, x0 - band : x0].flip(-1)
        inner = generated[:, :, y0:y1, x0 : x0 + band]
        return outer, inner
    if side == "right":
        band = min(int(band_width), width - x1, x1 - x0)
        if band <= 0:
            return None, None
        outer = reference[:, :, y0:y1, x1 : x1 + band]
        inner = generated[:, :, y0:y1, x1 - band : x1].flip(-1)
        return outer, inner
    if side == "top":
        band = min(int(band_width), y0, y1 - y0)
        if band <= 0:
            return None, None
        outer = reference[:, :, y0 - band : y0, x0:x1].flip(-2)
        inner = generated[:, :, y0 : y0 + band, x0:x1]
        return outer, inner
    if side == "bottom":
        band = min(int(band_width), height - y1, y1 - y0)
        if band <= 0:
            return None, None
        outer = reference[:, :, y1 : y1 + band, x0:x1]
        inner = generated[:, :, y1 - band : y1, x0:x1].flip(-2)
        return outer, inner
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


def _place_side_profile_delta(
    profile: torch.Tensor,
    bbox: tuple[int, int, int, int],
    side: str,
    band_width: int,
    full_shape: tuple[int, int, int, int],
) -> torch.Tensor:
    batch, channels, height, width = full_shape
    target = torch.zeros(batch, channels, height, width, device=profile.device, dtype=profile.dtype)
    x0, y0, x1, y1 = bbox
    band = max(int(band_width), 1)
    if side == "left":
        target[:, :, y0:y1, x0 : x0 + band] = profile.expand(-1, -1, -1, band)
    elif side == "right":
        target[:, :, y0:y1, x1 - band : x1] = profile.expand(-1, -1, -1, band)
    elif side == "top":
        target[:, :, y0 : y0 + band, x0:x1] = profile.expand(-1, -1, band, -1)
    elif side == "bottom":
        target[:, :, y1 - band : y1, x0:x1] = profile.expand(-1, -1, band, -1)
    else:
        raise ValueError(f"unsupported side: {side}")
    return target


def apply_neighbor_tone_match(
    reference_rgb: torch.Tensor,
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
    if reference_rgb.ndim != 4 or generated_rgb.ndim != 4:
        raise ValueError("reference_rgb and generated_rgb must be BCHW")
    if reference_rgb.shape != generated_rgb.shape:
        raise ValueError("reference_rgb and generated_rgb must have the same shape")

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
    generated_yuv = _rgb_to_yuv(generated_rgb)
    channel_scale = reference_yuv.new_tensor([float(luma_strength), float(chroma_strength), float(chroma_strength)]).view(1, 3, 1, 1)

    side_deltas: dict[str, torch.Tensor] = {}
    per_side_meta: dict[str, dict] = {}
    for side in sides:
        outer, inner = _extract_side_pair(reference_yuv, generated_yuv, bbox, side, int(inner_width))
        if outer is None or inner is None:
            continue
        delta = outer - inner
        profile = _reduce_side_delta(delta, side)
        profile = _lowpass_side_profile(
            profile,
            side=side,
            original_strip_shape=(delta.shape[-2], delta.shape[-1]),
            short_side=int(downsample_short_side),
        )
        placed = _place_side_profile_delta(profile * channel_scale, bbox, side, int(inner_width), generated_yuv.shape)
        side_deltas[side] = placed
        per_side_meta[side] = {
            "band_shape": [int(v) for v in delta.shape[-2:]],
            "profile_shape": [int(v) for v in profile.shape[-2:]],
            "mean_abs_delta": float(profile.abs().mean().item()),
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

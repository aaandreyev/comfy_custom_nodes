from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from ..strip_ops import mask_bbox
from .merge_bands import merge_side_deltas


# ---------------------------------------------------------------------------
# Colour space helpers
# ---------------------------------------------------------------------------

def _srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    safe = x.clamp_min(0.04045)
    return torch.where(x <= 0.04045, x / 12.92, ((safe + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    safe = x.clamp_min(0.0031308)
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * safe ** (1.0 / 2.4) - 0.055)


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


# ---------------------------------------------------------------------------
# YUV quantisation
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# LUT: build, fill empty bins, lookup
# ---------------------------------------------------------------------------

def _fill_lut_nn(correction: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Fill empty LUT bins via iterative nearest-neighbour expansion from valid bins."""
    if valid.all():
        return correction

    B, nb = correction.shape[0], correction.shape[1]

    # [B, 3, bins, bins, bins] — work in float32
    filled = correction.float().permute(0, 4, 1, 2, 3)
    cur_valid = valid.float().unsqueeze(1)  # [B, 1, bins, bins, bins]

    # Zero invalid bins so they don't bias the neighbour averages
    filled = filled * cur_valid

    for _ in range(nb * 3):
        if cur_valid.min() > 0.5:
            break
        expanded = (F.max_pool3d(cur_valid, kernel_size=3, stride=1, padding=1) > 0.5).float()
        newly = (expanded - cur_valid).clamp_min(0.0)
        if newly.sum() < 0.5:
            break
        # count_include_pad=True (default): div by 27 always → multiply back gives correct sum
        ch_count = F.avg_pool3d(cur_valid, 3, 1, 1) * 27.0  # [B, 1, ...]
        ch_sum = F.avg_pool3d(filled, 3, 1, 1) * 27.0       # [B, 3, ...]
        new_val = ch_sum / ch_count.clamp_min(1e-8)
        filled = torch.where((newly > 0.5).expand_as(filled), new_val, filled)
        cur_valid = expanded

    return filled.permute(0, 2, 3, 4, 1).to(correction.dtype)  # [B, bins, bins, bins, 3]


def _build_delta_lookup(
    drift_strip: torch.Tensor,
    reference_strip: torch.Tensor,
    *,
    bins: int = 32,
    mode: str = "hybrid",
) -> dict[str, torch.Tensor]:
    """
    Build a per-YUV-bin correction lookup from a pair of outer strips.

    correction_mode:
      "additive"      — ref_mean - drift_mean per bin (classic delta)
      "multiplicative"— ref_mean / drift_mean per bin (ratio, handles exposure)
      "hybrid"        — multiplicative luma + additive chroma (best default)

    Stored correction per bin:
      additive      → (delta_y, delta_u, delta_v)
      multiplicative→ (ratio_y, ratio_u, ratio_v)
      hybrid        → (ratio_y, delta_u, delta_v)
    """
    device = drift_strip.device
    orig_dtype = drift_strip.dtype
    drift = drift_strip.float()
    reference = reference_strip.float()

    B, _, H, W = drift.shape
    bins = max(int(bins), 2)

    qy, qu, qv = _quantize_yuv(drift, bins)  # each [B, H, W]

    # Vectorised batch+spatial flat index (no Python loop over batch)
    b_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, H, W)
    flat_idx = (b_idx * bins ** 3 + qy * bins ** 2 + qu * bins + qv).reshape(-1)

    total = B * bins ** 3
    counts = torch.zeros(total, dtype=torch.float32, device=device)
    drift_sum = torch.zeros(total, 3, dtype=torch.float32, device=device)
    ref_sum = torch.zeros(total, 3, dtype=torch.float32, device=device)

    ones = torch.ones(flat_idx.shape[0], dtype=torch.float32, device=device)
    drift_flat = drift.permute(0, 2, 3, 1).reshape(-1, 3)
    ref_flat = reference.permute(0, 2, 3, 1).reshape(-1, 3)

    counts.index_add_(0, flat_idx, ones)
    drift_sum.index_add_(0, flat_idx, drift_flat)
    ref_sum.index_add_(0, flat_idx, ref_flat)

    counts = counts.view(B, bins, bins, bins)
    drift_sum = drift_sum.view(B, bins, bins, bins, 3)
    ref_sum = ref_sum.view(B, bins, bins, bins, 3)

    valid = counts > 0
    c = counts.unsqueeze(-1).clamp_min(1e-8)
    drift_mean = drift_sum / c
    ref_mean = ref_sum / c

    if mode == "additive":
        correction = ref_mean - drift_mean
        identity = torch.zeros_like(correction)
    elif mode == "multiplicative":
        correction = ref_mean / drift_mean.clamp_min(1e-6)
        identity = torch.ones_like(correction)
    else:  # hybrid
        luma_ratio = ref_mean[..., :1] / drift_mean[..., :1].clamp_min(1e-6)
        chroma_delta = ref_mean[..., 1:] - drift_mean[..., 1:]
        correction = torch.cat([luma_ratio, chroma_delta], dim=-1)
        identity = torch.zeros_like(correction)
        identity[..., 0] = 1.0  # luma identity = ratio 1

    # Pre-fill empty bins with identity so bins unreachable by NN expansion are neutral
    correction = torch.where(valid.unsqueeze(-1), correction, identity)

    # Nearest-neighbour fill: propagate valid-bin values into empty neighbours
    correction = _fill_lut_nn(correction, valid)

    return {
        "correction": correction.to(orig_dtype),
        "bins": torch.tensor(bins, device=device),
        "mode": mode,
    }


def _lookup_delta(
    generated_band: torch.Tensor,
    lookup: dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    Look up per-pixel additive delta from the pre-built LUT.
    Always returns a tensor to ADD to generated YUV.
    """
    bins = int(lookup["bins"].item())
    mode = lookup["mode"]
    correction = lookup["correction"]  # [B, bins, bins, bins, 3]

    qy, qu, qv = _quantize_yuv(generated_band, bins)
    B = generated_band.shape[0]

    b_idx = torch.arange(B, device=generated_band.device).view(B, 1, 1)
    looked_up = correction[b_idx, qy, qu, qv]  # [B, H, W, 3]
    looked_up = looked_up.permute(0, 3, 1, 2)  # [B, 3, H, W]

    if mode == "additive":
        return looked_up
    elif mode == "multiplicative":
        # corrected = gen * ratio  →  delta = gen * (ratio - 1)
        return generated_band * (looked_up - 1.0)
    else:  # hybrid
        luma_delta = generated_band[:, :1] * (looked_up[:, :1] - 1.0)
        chroma_delta = looked_up[:, 1:]
        return torch.cat([luma_delta, chroma_delta], dim=1)


# ---------------------------------------------------------------------------
# Strip extraction and delta placement
# ---------------------------------------------------------------------------

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
        target[:, :, y0:y1, x1 - delta.shape[-1] : x1] = delta
    elif side == "top":
        target[:, :, y0 : y0 + delta.shape[-2], x0:x1] = delta
    elif side == "bottom":
        target[:, :, y1 - delta.shape[-2] : y1, x0:x1] = delta
    else:
        raise ValueError(f"unsupported side: {side}")
    return target


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

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
    luma_strength: float,
    chroma_strength: float,
    bins: int = 32,
    correction_mode: str = "hybrid",
    color_space: str = "linear",
    corner_px: float | None = None,
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

    # Optionally linearise before YUV conversion
    if color_space == "srgb":
        ref_lin = _srgb_to_linear(reference_rgb)
        drift_lin = _srgb_to_linear(drift_source_rgb)
        gen_lin = _srgb_to_linear(generated_rgb)
    else:
        ref_lin = reference_rgb
        drift_lin = drift_source_rgb
        gen_lin = generated_rgb

    reference_yuv = _rgb_to_yuv(ref_lin)
    drift_source_yuv = _rgb_to_yuv(drift_lin)
    generated_yuv = _rgb_to_yuv(gen_lin)
    channel_scale = reference_yuv.new_tensor(
        [float(luma_strength), float(chroma_strength), float(chroma_strength)]
    ).view(1, 3, 1, 1)

    side_deltas: dict[str, torch.Tensor] = {}
    per_side_meta: dict[str, dict] = {}
    for side in sides:
        reference_outer = _extract_outer_side_strip(reference_yuv, bbox, side, int(inner_width))
        drift_outer = _extract_outer_side_strip(drift_source_yuv, bbox, side, int(inner_width))
        generated_inner = _extract_inner_side_band(generated_yuv, bbox, side, int(inner_width))
        if reference_outer is None or drift_outer is None or generated_inner is None:
            continue
        lookup = _build_delta_lookup(drift_outer, reference_outer, bins=int(bins), mode=correction_mode)
        delta_band = _lookup_delta(generated_inner, lookup) * channel_scale
        placed = _place_side_delta(delta_band, bbox, side, generated_yuv.shape)
        side_deltas[side] = placed
        per_side_meta[side] = {
            "band_shape": [int(v) for v in generated_inner.shape[-2:]],
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
        corner_px=corner_px,
    )
    corrected_yuv = generated_yuv + merged_delta
    corrected_lin = _yuv_to_rgb(corrected_yuv).clamp(0.0, 1.0)

    if color_space == "srgb":
        corrected_rgb = _linear_to_srgb(corrected_lin)
    else:
        corrected_rgb = corrected_lin

    return corrected_rgb, {
        "reason": "applied",
        "bbox": bbox,
        "side_deltas": side_deltas,
        "weights": weights,
        "merged_delta": merged_delta,
        "per_side": per_side_meta,
    }


# ---------------------------------------------------------------------------
# Debug output
# ---------------------------------------------------------------------------

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

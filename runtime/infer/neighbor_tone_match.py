from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import distance_transform_edt

from ..strip_ops import mask_bbox
from .merge_bands import merge_side_deltas
from .seam_latent_anchor import _normalize_mask_like, _parse_present_positions, SIDE_TOPOLOGY


def resolve_compute_device() -> torch.device | None:
    """ComfyUI's torch device when available and non-CPU, else None.

    Tone match math is pure tensor work; running it on the sampler GPU instead
    of the CPU-resident IMAGE tensors is a large win. Callers move inputs to
    this device and move the result back.
    """
    try:
        from comfy import model_management
    except Exception:
        return None
    try:
        device = model_management.get_torch_device()
    except Exception:
        return None
    if device is None or device.type == "cpu":
        return None
    return device


# ---------------------------------------------------------------------------
# Colour space helpers
# ---------------------------------------------------------------------------

def _srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    safe = x.clamp_min(0.04045)
    return torch.where(x <= 0.04045, x / 12.92, ((safe + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    safe = x.clamp_min(0.0031308)
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * safe ** (1.0 / 2.4) - 0.055)


def _yuv_coefficients(matrix: str) -> tuple[tuple[float, float, float], float, float]:
    key = str(matrix).lower()
    if key == "bt601":
        return (0.299, 0.587, 0.114), 0.436, 0.615
    if key == "bt709":
        return (0.2126, 0.7152, 0.0722), 0.436, 0.615
    raise ValueError(f"unsupported yuv_matrix: {matrix}")


def _rgb_to_yuv(rgb: torch.Tensor, *, matrix: str = "bt709") -> torch.Tensor:
    (kr, kg, kb), umax, vmax = _yuv_coefficients(matrix)
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    y = kr * r + kg * g + kb * b
    u = umax * (b - y) / max(1.0 - kb, 1e-8)
    v = vmax * (r - y) / max(1.0 - kr, 1e-8)
    return torch.cat([y, u, v], dim=1)


def _yuv_to_rgb(yuv: torch.Tensor, *, matrix: str = "bt709") -> torch.Tensor:
    (kr, kg, kb), umax, vmax = _yuv_coefficients(matrix)
    y, u, v = yuv[:, 0:1], yuv[:, 1:2], yuv[:, 2:3]
    r = y + v * ((1.0 - kr) / vmax)
    b = y + u * ((1.0 - kb) / umax)
    g = (y - kr * r - kb * b) / max(kg, 1e-8)
    return torch.cat([r, g, b], dim=1)


# ---------------------------------------------------------------------------
# YUV quantisation
# ---------------------------------------------------------------------------

def _normalize_yuv_for_lookup(yuv: torch.Tensor, *, matrix: str = "bt709") -> torch.Tensor:
    _, umax, vmax = _yuv_coefficients(matrix)
    out = yuv.clone()
    out[:, 0:1] = out[:, 0:1].clamp(0.0, 1.0)
    out[:, 1:2] = ((out[:, 1:2] + umax) / (2.0 * umax)).clamp(0.0, 1.0)
    out[:, 2:3] = ((out[:, 2:3] + vmax) / (2.0 * vmax)).clamp(0.0, 1.0)
    return out


def _quantize_yuv(yuv: torch.Tensor, bins: int, *, matrix: str = "bt709") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bins = max(int(bins), 2)
    norm = _normalize_yuv_for_lookup(yuv, matrix=matrix)
    q = (norm * float(bins - 1)).round().long().clamp(0, bins - 1)
    return q[:, 0], q[:, 1], q[:, 2]


def _safe_ratio(
    numer: torch.Tensor,
    denom: torch.Tensor,
    *,
    eps: float = 1e-4,
    min_ratio: float = 0.5,
    max_ratio: float = 2.0,
    signed: bool = False,
) -> torch.Tensor:
    if signed:
        safe_denom = denom.sign() * denom.abs().clamp_min(eps)
        ratio = numer / safe_denom
        ratio = ratio.clamp(-max_ratio, max_ratio)
    else:
        ratio = numer / denom.clamp_min(eps)
        ratio = ratio.clamp(min_ratio, max_ratio)
    near_zero = numer.abs() < eps
    if signed:
        near_zero = near_zero & (denom.abs() < eps)
    else:
        near_zero = near_zero & (denom < eps)
    return torch.where(near_zero, torch.ones_like(ratio), ratio)


# ---------------------------------------------------------------------------
# LUT: build, fill empty bins, lookup
# ---------------------------------------------------------------------------

def _nearest_valid_fill(
    values: torch.Tensor,
    valid: torch.Tensor,
    global_value: torch.Tensor,
) -> torch.Tensor:
    """Fill invalid grid cells with the value of the exact nearest valid cell.

    Single-pass EDT replaces the previous O(bins^4) iterative dilation.
    values: [B, *grid, C]; valid: [B, *grid]; global_value: [B, C] is used only
    when a batch element has no valid cells at all.
    """
    if bool(valid.all()):
        return values
    out = values.clone()
    grid_ndim = valid.ndim - 1
    for b in range(values.shape[0]):
        v = valid[b]
        if not bool(v.any()):
            out[b] = global_value[b].to(values.dtype).view(*([1] * grid_ndim), -1)
            continue
        invalid_np = (~v).detach().cpu().numpy()
        nearest = distance_transform_edt(invalid_np, return_indices=True, return_distances=False)
        index = tuple(
            torch.from_numpy(np.ascontiguousarray(axis)).long().to(values.device)
            for axis in nearest
        )
        out[b] = values[b][index]
    return out


def _global_correction_from_samples(
    drift: torch.Tensor,
    reference: torch.Tensor,
    *,
    mode: str,
) -> torch.Tensor:
    drift_mean = drift.mean(dim=-1)
    ref_mean = reference.mean(dim=-1)
    if mode == "additive":
        return ref_mean - drift_mean
    if mode == "multiplicative":
        luma_ratio = _safe_ratio(ref_mean[:, :1], drift_mean[:, :1])
        chroma_ratio = _safe_ratio(ref_mean[:, 1:], drift_mean[:, 1:], signed=True)
        return torch.cat([luma_ratio, chroma_ratio], dim=1)
    luma_ratio = _safe_ratio(ref_mean[:, :1], drift_mean[:, :1])
    chroma_delta = ref_mean[:, 1:] - drift_mean[:, 1:]
    return torch.cat([luma_ratio, chroma_delta], dim=1)


def _build_delta_lookup(
    drift_samples: torch.Tensor,
    reference_samples: torch.Tensor,
    *,
    bins: int = 32,
    mode: str = "hybrid",
    lut_mode: str = "3d",
    matrix: str = "bt709",
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
    device = drift_samples.device
    orig_dtype = drift_samples.dtype
    drift = drift_samples.float()
    reference = reference_samples.float()

    B, _, N = drift.shape
    bins = max(int(bins), 2)
    global_correction = _global_correction_from_samples(drift, reference, mode=mode)

    if lut_mode == "2d_luma_curve":
        norm = _normalize_yuv_for_lookup(drift.view(B, 3, 1, N), matrix=matrix).view(B, 3, N)
        qy = (norm[:, 0] * float(bins - 1)).round().long().clamp(0, bins - 1)
        qu = (norm[:, 1] * float(bins - 1)).round().long().clamp(0, bins - 1)
        qv = (norm[:, 2] * float(bins - 1)).round().long().clamp(0, bins - 1)

        total_luma = B * bins
        total_chroma = B * bins * bins
        b_flat = torch.arange(B, device=device).view(B, 1).expand(B, N)
        luma_idx = (b_flat * bins + qy).reshape(-1)
        chroma_idx = (b_flat * bins * bins + qu * bins + qv).reshape(-1)
        counts_luma = torch.zeros(total_luma, dtype=torch.float32, device=device)
        counts_chroma = torch.zeros(total_chroma, dtype=torch.float32, device=device)
        drift_luma_sum = torch.zeros(total_luma, 1, dtype=torch.float32, device=device)
        ref_luma_sum = torch.zeros(total_luma, 1, dtype=torch.float32, device=device)
        drift_chroma_sum = torch.zeros(total_chroma, 2, dtype=torch.float32, device=device)
        ref_chroma_sum = torch.zeros(total_chroma, 2, dtype=torch.float32, device=device)
        ones = torch.ones(B * N, dtype=torch.float32, device=device)
        drift_luma = drift[:, :1].permute(0, 2, 1).reshape(-1, 1)
        ref_luma = reference[:, :1].permute(0, 2, 1).reshape(-1, 1)
        drift_chroma = drift[:, 1:].permute(0, 2, 1).reshape(-1, 2)
        ref_chroma = reference[:, 1:].permute(0, 2, 1).reshape(-1, 2)
        counts_luma.index_add_(0, luma_idx, ones)
        counts_chroma.index_add_(0, chroma_idx, ones)
        drift_luma_sum.index_add_(0, luma_idx, drift_luma)
        ref_luma_sum.index_add_(0, luma_idx, ref_luma)
        drift_chroma_sum.index_add_(0, chroma_idx, drift_chroma)
        ref_chroma_sum.index_add_(0, chroma_idx, ref_chroma)
        counts_luma = counts_luma.view(B, bins)
        counts_chroma = counts_chroma.view(B, bins, bins)
        drift_luma_mean = drift_luma_sum.view(B, bins, 1) / counts_luma.unsqueeze(-1).clamp_min(1e-8)
        ref_luma_mean = ref_luma_sum.view(B, bins, 1) / counts_luma.unsqueeze(-1).clamp_min(1e-8)
        drift_chroma_mean = drift_chroma_sum.view(B, bins, bins, 2) / counts_chroma.unsqueeze(-1).clamp_min(1e-8)
        ref_chroma_mean = ref_chroma_sum.view(B, bins, bins, 2) / counts_chroma.unsqueeze(-1).clamp_min(1e-8)
        valid_luma = counts_luma > 0
        valid_chroma = counts_chroma > 0
        luma_curve = _safe_ratio(ref_luma_mean, drift_luma_mean) if mode != "additive" else (ref_luma_mean - drift_luma_mean)
        chroma_lookup = ref_chroma_mean - drift_chroma_mean if mode != "multiplicative" else _safe_ratio(ref_chroma_mean, drift_chroma_mean, signed=True)
        luma_curve = _nearest_valid_fill(luma_curve, valid_luma, global_correction[:, :1]).to(orig_dtype)
        chroma_lookup = _nearest_valid_fill(chroma_lookup, valid_chroma, global_correction[:, 1:]).to(orig_dtype)
        return {
            "lut_mode": lut_mode,
            "luma_curve": luma_curve,
            "chroma_lookup": chroma_lookup,
            "bins": torch.tensor(bins, device=device),
            "mode": mode,
            "matrix": matrix,
            "global_correction": global_correction.to(orig_dtype),
        }

    norm = _normalize_yuv_for_lookup(drift.view(B, 3, 1, N), matrix=matrix).view(B, 3, N)
    qy = (norm[:, 0] * float(bins - 1)).round().long().clamp(0, bins - 1)
    qu = (norm[:, 1] * float(bins - 1)).round().long().clamp(0, bins - 1)
    qv = (norm[:, 2] * float(bins - 1)).round().long().clamp(0, bins - 1)
    b_idx = torch.arange(B, device=device).view(B, 1).expand(B, N)
    flat_idx = (b_idx * bins ** 3 + qy * bins ** 2 + qu * bins + qv).reshape(-1)

    total = B * bins ** 3
    counts = torch.zeros(total, dtype=torch.float32, device=device)
    drift_sum = torch.zeros(total, 3, dtype=torch.float32, device=device)
    ref_sum = torch.zeros(total, 3, dtype=torch.float32, device=device)

    ones = torch.ones(B * N, dtype=torch.float32, device=device)
    drift_flat = drift.permute(0, 2, 1).reshape(-1, 3)
    ref_flat = reference.permute(0, 2, 1).reshape(-1, 3)

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
    elif mode == "multiplicative":
        luma_ratio = _safe_ratio(ref_mean[..., :1], drift_mean[..., :1])
        chroma_ratio = _safe_ratio(ref_mean[..., 1:], drift_mean[..., 1:], signed=True)
        correction = torch.cat([luma_ratio, chroma_ratio], dim=-1)
    else:  # hybrid
        luma_ratio = _safe_ratio(ref_mean[..., :1], drift_mean[..., :1])
        chroma_delta = ref_mean[..., 1:] - drift_mean[..., 1:]
        correction = torch.cat([luma_ratio, chroma_delta], dim=-1)

    correction = _nearest_valid_fill(correction, valid, global_correction)

    return {
        "correction": correction.to(orig_dtype),
        "bins": torch.tensor(bins, device=device),
        "mode": mode,
        "lut_mode": lut_mode,
        "matrix": matrix,
        "global_correction": global_correction.to(orig_dtype),
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
    lut_mode = lookup.get("lut_mode", "3d")
    matrix = str(lookup.get("matrix", "bt709"))
    norm = _normalize_yuv_for_lookup(generated_band, matrix=matrix) * float(bins - 1)
    lo = norm.floor().long().clamp(0, bins - 1)
    hi = (lo + 1).clamp(0, bins - 1)
    frac = (norm - lo.float()).clamp(0.0, 1.0)
    y0, u0, v0 = lo[:, 0], lo[:, 1], lo[:, 2]
    y1, u1, v1 = hi[:, 0], hi[:, 1], hi[:, 2]
    fy, fu, fv = frac[:, 0:1], frac[:, 1:2], frac[:, 2:3]
    B = generated_band.shape[0]
    b_idx = torch.arange(B, device=generated_band.device).view(B, 1, 1)

    if lut_mode == "2d_luma_curve":
        luma_curve = lookup["luma_curve"]
        chroma_lookup = lookup["chroma_lookup"]

        def gather_luma(yy: torch.Tensor) -> torch.Tensor:
            return luma_curve[b_idx, yy].permute(0, 3, 1, 2)

        def gather_chroma(uu: torch.Tensor, vv: torch.Tensor) -> torch.Tensor:
            return chroma_lookup[b_idx, uu, vv].permute(0, 3, 1, 2)

        looked_luma = gather_luma(y0) * (1.0 - fy) + gather_luma(y1) * fy
        c00 = gather_chroma(u0, v0)
        c01 = gather_chroma(u0, v1)
        c10 = gather_chroma(u1, v0)
        c11 = gather_chroma(u1, v1)
        looked_chroma = (
            c00 * ((1.0 - fu) * (1.0 - fv)) +
            c01 * ((1.0 - fu) * fv) +
            c10 * (fu * (1.0 - fv)) +
            c11 * (fu * fv)
        )
        looked_up = torch.cat([looked_luma, looked_chroma], dim=1)
    else:
        correction = lookup["correction"]

        def gather(yy: torch.Tensor, uu: torch.Tensor, vv: torch.Tensor) -> torch.Tensor:
            return correction[b_idx, yy, uu, vv].permute(0, 3, 1, 2)

        c000 = gather(y0, u0, v0)
        c001 = gather(y0, u0, v1)
        c010 = gather(y0, u1, v0)
        c011 = gather(y0, u1, v1)
        c100 = gather(y1, u0, v0)
        c101 = gather(y1, u0, v1)
        c110 = gather(y1, u1, v0)
        c111 = gather(y1, u1, v1)
        fy0 = 1.0 - fy
        fu0 = 1.0 - fu
        fv0 = 1.0 - fv
        looked_up = (
            c000 * (fy0 * fu0 * fv0) +
            c001 * (fy0 * fu0 * fv) +
            c010 * (fy0 * fu * fv0) +
            c011 * (fy0 * fu * fv) +
            c100 * (fy * fu0 * fv0) +
            c101 * (fy * fu0 * fv) +
            c110 * (fy * fu * fv0) +
            c111 * (fy * fu * fv)
        )

    if mode == "additive":
        return looked_up
    elif mode == "multiplicative":
        # corrected = gen * ratio  →  delta = gen * (ratio - 1)
        return generated_band * (looked_up - 1.0)
    else:  # hybrid
        luma_delta = generated_band[:, :1] * (looked_up[:, :1] - 1.0)
        chroma_delta = looked_up[:, 1:]
        return torch.cat([luma_delta, chroma_delta], dim=1)


def _gather_outer_samples(
    image: torch.Tensor,
    bbox: tuple[int, int, int, int],
    sides: list[str],
    band_width: int,
) -> torch.Tensor | None:
    strips: list[torch.Tensor] = []
    for side in sides:
        strip = _extract_outer_side_strip(image, bbox, side, band_width)
        if strip is not None:
            strips.append(strip.flatten(2))
    if not strips:
        return None
    return torch.cat(strips, dim=-1)


def _gather_outer_samples_per_element(
    image: torch.Tensor,
    soft_mask: torch.Tensor,
    sides: list[str],
    band_width: int,
    union_bbox: tuple[int, int, int, int],
) -> torch.Tensor | None:
    """Gather outer strips using per-element bboxes so each element only samples near its own mask."""
    B = image.shape[0]
    per_elem: list[torch.Tensor | None] = []
    min_n: int | None = None

    for b in range(B):
        m_b = (soft_mask[b : b + 1] > 1e-3).to(dtype=image.dtype)
        try:
            bbox_b = mask_bbox(m_b)
        except RuntimeError:
            bbox_b = union_bbox
        strips: list[torch.Tensor] = []
        for side in sides:
            strip = _extract_outer_side_strip(image[b : b + 1], bbox_b, side, band_width)
            if strip is not None:
                strips.append(strip.flatten(2))
        if strips:
            cat = torch.cat(strips, dim=-1)
            per_elem.append(cat)
            min_n = cat.shape[-1] if min_n is None else min(min_n, cat.shape[-1])
        else:
            per_elem.append(None)

    valid = [s for s in per_elem if s is not None]
    if not valid or not min_n:
        return None

    placeholder = valid[0][:, :, :min_n]
    result = [t[:, :, :min_n] if t is not None else placeholder for t in per_elem]
    return torch.cat(result, dim=0)


def _gaussian_kernel_1d(sigma: float, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    radius = max(1, int(math.ceil(sigma * 3.0)))
    coords = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    kernel = torch.exp(-(coords.square()) / max(2.0 * sigma * sigma, 1e-8))
    kernel = kernel / kernel.sum().clamp_min(1e-8)
    return kernel.to(dtype=dtype)


def _gaussian_blur_band(delta_band: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0.0:
        return delta_band
    kernel = _gaussian_kernel_1d(sigma, delta_band.dtype, delta_band.device)
    pad = kernel.numel() // 2
    ky = kernel.view(1, 1, -1, 1)
    kx = kernel.view(1, 1, 1, -1)
    channels = delta_band.shape[1]
    out = F.conv2d(F.pad(delta_band, (0, 0, pad, pad), mode="replicate"), ky.expand(channels, 1, -1, 1), groups=channels)
    out = F.conv2d(F.pad(out, (pad, pad, 0, 0), mode="replicate"), kx.expand(channels, 1, 1, -1), groups=channels)
    return out


def _compress_to_unit_gamut(rgb: torch.Tensor, neutral: torch.Tensor) -> torch.Tensor:
    delta = rgb - neutral
    scale = torch.ones_like(neutral)
    positive = delta > 1e-8
    negative = delta < -1e-8
    if bool(positive.any()):
        scale = torch.minimum(
            scale,
            torch.where(positive, (1.0 - neutral) / delta.clamp_min(1e-8), torch.ones_like(delta)).amin(dim=1, keepdim=True),
        )
    if bool(negative.any()):
        scale = torch.minimum(
            scale,
            torch.where(negative, neutral / (-delta).clamp_min(1e-8), torch.ones_like(delta)).amin(dim=1, keepdim=True),
        )
    return (neutral + delta * scale.clamp(0.0, 1.0)).clamp(0.0, 1.0)


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
    topology_mask: torch.Tensor | None = None,
    *,
    inner_width: int,
    inner_flat_top_px: int = 0,
    process_left: bool,
    process_right: bool,
    process_top: bool,
    process_bottom: bool,
    luma_strength: float,
    chroma_strength: float,
    u_strength: float | None = None,
    v_strength: float | None = None,
    bins: int = 32,
    correction_mode: str = "hybrid",
    color_space: str = "srgb",
    corner_px: float | None = None,
    outer_band_px: int | None = None,
    lut_mode: str = "3d",
    yuv_matrix: str = "bt709",
    delta_smoothing_sigma: float = 2.0,
) -> tuple[torch.Tensor, dict]:
    if reference_rgb.ndim != 4 or drift_source_rgb.ndim != 4 or generated_rgb.ndim != 4:
        raise ValueError("reference_rgb, drift_source_rgb, and generated_rgb must be BCHW")
    if reference_rgb.shape != generated_rgb.shape or reference_rgb.shape != drift_source_rgb.shape:
        raise ValueError("reference_rgb, drift_source_rgb, and generated_rgb must have the same shape")

    image_dtype = reference_rgb.dtype
    u_strength = float(chroma_strength if u_strength is None else u_strength)
    v_strength = float(chroma_strength if v_strength is None else v_strength)
    mask = mask.to(dtype=image_dtype)
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    elif mask.ndim != 4:
        raise ValueError("mask must be [B,H,W] or [B,1,H,W]")
    if mask.shape[-2:] != reference_rgb.shape[-2:]:
        interp_mode = "bilinear" if mask.dtype.is_floating_point else "nearest"
        if interp_mode == "bilinear":
            mask = F.interpolate(mask, size=reference_rgb.shape[-2:], mode=interp_mode, align_corners=False)
        else:
            mask = F.interpolate(mask, size=reference_rgb.shape[-2:], mode=interp_mode)
    soft_mask = mask.clamp(0.0, 1.0)
    if soft_mask.shape[0] == 1 and reference_rgb.shape[0] > 1:
        soft_mask = soft_mask.expand(reference_rgb.shape[0], -1, -1, -1)
    bbox = mask_bbox((soft_mask > 1e-3).to(dtype=image_dtype))
    x0, y0, x1, y1 = bbox
    present_positions = _parse_present_positions(
        topology_mask,
        bbox,
        reference_rgb.shape[-2:],
        device=reference_rgb.device,
        dtype=image_dtype,
    )
    sides = []
    if process_left and x0 > 0 and (not present_positions or SIDE_TOPOLOGY["left"] in present_positions):
        sides.append("left")
    if process_right and x1 < reference_rgb.shape[-1] and (not present_positions or SIDE_TOPOLOGY["right"] in present_positions):
        sides.append("right")
    if process_top and y0 > 0 and (not present_positions or SIDE_TOPOLOGY["top"] in present_positions):
        sides.append("top")
    if process_bottom and y1 < reference_rgb.shape[-2] and (not present_positions or SIDE_TOPOLOGY["bottom"] in present_positions):
        sides.append("bottom")
    if not sides:
        return generated_rgb, {
            "reason": "no_processable_sides",
            "side_deltas": {},
            "weights": {},
            "bbox": bbox,
            "present_positions": tuple(sorted(present_positions)),
            "topology_mask": _normalize_mask_like(topology_mask, reference_rgb.shape[-2:]) if topology_mask is not None else None,
        }

    # Work on a crop around the mask bbox: bbox + outer_band on each side covers
    # every pixel the correction can touch, so the result is identical to the
    # full-frame computation while YUV conversion, band merge, and gamut
    # compression all scale with the mask, not the frame.
    height, width = reference_rgb.shape[-2:]
    outer_width = int(inner_width if outer_band_px is None else outer_band_px)
    pad = max(outer_width, 1)
    cx0, cy0 = max(x0 - pad, 0), max(y0 - pad, 0)
    cx1, cy1 = min(x1 + pad, width), min(y1 + pad, height)
    bbox_c = (x0 - cx0, y0 - cy0, x1 - cx0, y1 - cy0)
    ref_crop = reference_rgb[:, :, cy0:cy1, cx0:cx1]
    drift_crop = drift_source_rgb[:, :, cy0:cy1, cx0:cx1]
    gen_crop = generated_rgb[:, :, cy0:cy1, cx0:cx1]
    mask_crop = soft_mask[:, :, cy0:cy1, cx0:cx1]

    # Optionally linearise before YUV conversion
    if color_space == "srgb":
        ref_lin = _srgb_to_linear(ref_crop)
        drift_lin = _srgb_to_linear(drift_crop)
        gen_lin = _srgb_to_linear(gen_crop)
    else:
        ref_lin = ref_crop
        drift_lin = drift_crop
        gen_lin = gen_crop

    reference_yuv = _rgb_to_yuv(ref_lin, matrix=yuv_matrix)
    drift_source_yuv = _rgb_to_yuv(drift_lin, matrix=yuv_matrix)
    generated_yuv = _rgb_to_yuv(gen_lin, matrix=yuv_matrix)
    channel_scale = reference_yuv.new_tensor(
        [float(luma_strength), u_strength, v_strength]
    ).view(1, 3, 1, 1)

    side_deltas: dict[str, torch.Tensor] = {}
    per_side_meta: dict[str, dict] = {}
    reference_outer_samples = _gather_outer_samples_per_element(reference_yuv, mask_crop, sides, outer_width, bbox_c)
    drift_outer_samples = _gather_outer_samples_per_element(drift_source_yuv, mask_crop, sides, outer_width, bbox_c)
    lookup = None
    if reference_outer_samples is not None and drift_outer_samples is not None:
        lookup = _build_delta_lookup(
            drift_outer_samples,
            reference_outer_samples,
            bins=int(bins),
            mode=correction_mode,
            lut_mode=lut_mode,
            matrix=yuv_matrix,
        )
    for side in sides:
        generated_inner = _extract_inner_side_band(generated_yuv, bbox_c, side, int(inner_width))
        if lookup is None or generated_inner is None:
            continue
        delta_band = _lookup_delta(generated_inner, lookup) * channel_scale
        delta_band = _gaussian_blur_band(delta_band, float(delta_smoothing_sigma))
        placed = _place_side_delta(delta_band, bbox_c, side, generated_yuv.shape)
        side_deltas[side] = placed
        per_side_meta[side] = {
            "band_shape": [int(v) for v in generated_inner.shape[-2:]],
            "mean_abs_delta": float(delta_band.abs().mean().item()),
        }

    if not side_deltas:
        return generated_rgb, {
            "reason": "no_valid_side_deltas",
            "side_deltas": {},
            "weights": {},
            "bbox": bbox,
            "present_positions": tuple(sorted(present_positions)),
        }

    merged_delta, weights = merge_side_deltas(
        side_deltas,
        mask_crop,
        bbox=bbox_c,
        inner_width=int(inner_width),
        flat_top_px=int(inner_flat_top_px),
        corner_px=corner_px,
        active_sides=set(side_deltas),
    )
    merged_delta = merged_delta.to(dtype=image_dtype)
    corrected_yuv = generated_yuv + merged_delta
    corrected_lin = _compress_to_unit_gamut(
        _yuv_to_rgb(corrected_yuv, matrix=yuv_matrix),
        corrected_yuv[:, :1].clamp(0.0, 1.0).expand(-1, 3, -1, -1),
    )

    if color_space == "srgb":
        corrected_crop = _linear_to_srgb(corrected_lin).clamp(0.0, 1.0)
    else:
        corrected_crop = corrected_lin
    corrected_crop = corrected_crop * mask_crop + gen_crop * (1.0 - mask_crop)
    corrected_rgb = generated_rgb.clone()
    corrected_rgb[:, :, cy0:cy1, cx0:cx1] = corrected_crop.to(dtype=image_dtype)

    return corrected_rgb, {
        "reason": "applied",
        "bbox": bbox,
        "crop_offset": (cx0, cy0),
        "present_positions": tuple(sorted(present_positions)),
        "side_deltas": side_deltas,
        "weights": weights,
        "merged_delta": merged_delta,
        "soft_mask": soft_mask,
        "lut_mode": lut_mode,
        "yuv_matrix": yuv_matrix,
        "per_side": per_side_meta,
    }


def _freeform_inner_weight(
    dist_in: np.ndarray,
    *,
    inner_width: int,
    flat_top_px: int,
) -> np.ndarray:
    band = max(int(inner_width), 1)
    flat = min(max(int(flat_top_px), 0), band)
    radius = np.maximum(dist_in - 1.0, 0.0).astype(np.float32)
    if flat <= 0:
        t = np.clip(radius / float(max(band, 1)), 0.0, 1.0)
        return 0.5 * (1.0 + np.cos(np.pi * t))
    fade = max(float(band - flat), 1.0)
    t = np.clip((radius - float(flat)) / fade, 0.0, 1.0)
    weight = np.where(radius <= float(flat), 1.0, 0.5 * (1.0 + np.cos(np.pi * t)))
    return weight.astype(np.float32)


def apply_freeform_neighbor_tone_match(
    reference_rgb: torch.Tensor,
    image_rgb: torch.Tensor,
    mask: torch.Tensor,
    *,
    inner_width: int,
    inner_flat_top_px: int = 0,
    luma_strength: float,
    chroma_strength: float,
    u_strength: float | None = None,
    v_strength: float | None = None,
    bins: int = 32,
    correction_mode: str = "hybrid",
    color_space: str = "srgb",
    outer_band_px: int | None = None,
    lut_mode: str = "3d",
    yuv_matrix: str = "bt709",
    delta_smoothing_sigma: float = 2.0,
) -> tuple[torch.Tensor, dict]:
    if reference_rgb.ndim != 4 or image_rgb.ndim != 4:
        raise ValueError("reference_rgb and image_rgb must be BCHW")
    if reference_rgb.shape != image_rgb.shape:
        raise ValueError("reference_rgb and image_rgb must have the same shape")

    image_dtype = reference_rgb.dtype
    u_strength = float(chroma_strength if u_strength is None else u_strength)
    v_strength = float(chroma_strength if v_strength is None else v_strength)

    mask = mask.to(dtype=image_dtype)
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    elif mask.ndim != 4:
        raise ValueError("mask must be [B,H,W] or [B,1,H,W]")
    if mask.shape[-2:] != reference_rgb.shape[-2:]:
        interp_mode = "bilinear" if mask.dtype.is_floating_point else "nearest"
        if interp_mode == "bilinear":
            mask = F.interpolate(mask, size=reference_rgb.shape[-2:], mode=interp_mode, align_corners=False)
        else:
            mask = F.interpolate(mask, size=reference_rgb.shape[-2:], mode=interp_mode)
    soft_mask = mask.clamp(0.0, 1.0)
    if soft_mask.shape[0] == 1 and reference_rgb.shape[0] > 1:
        soft_mask = soft_mask.expand(reference_rgb.shape[0], -1, -1, -1)
    bbox = mask_bbox((soft_mask > 1e-3).to(dtype=image_dtype))

    channel_scale = reference_rgb.new_tensor([float(luma_strength), u_strength, v_strength]).view(1, 3, 1, 1)
    corrected = image_rgb.clone()
    debug_items: list[dict] = []
    outer_width = int(inner_width if outer_band_px is None else outer_band_px)
    margin = int(math.ceil(delta_smoothing_sigma * 3)) if delta_smoothing_sigma > 0.0 else 0
    # Crop pad covers the outer donor band, the blur margin, and one pixel of
    # guaranteed outside ring so both distance transforms stay exact on the crop.
    pad = max(outer_width, margin, 1) + 1
    height, width = reference_rgb.shape[-2:]

    for idx in range(reference_rgb.shape[0]):
        active = soft_mask[idx, 0] > 1e-3
        if not bool(active.any()):
            debug_items.append({"reason": "empty_mask", "bbox": bbox})
            continue
        ys, xs = torch.where(active)
        cy0 = max(int(ys.min()) - pad, 0)
        cy1 = min(int(ys.max()) + 1 + pad, height)
        cx0 = max(int(xs.min()) - pad, 0)
        cx1 = min(int(xs.max()) + 1 + pad, width)
        ref_crop = reference_rgb[idx : idx + 1, :, cy0:cy1, cx0:cx1]
        img_crop = image_rgb[idx : idx + 1, :, cy0:cy1, cx0:cx1]
        mask_crop = soft_mask[idx : idx + 1, :, cy0:cy1, cx0:cx1]
        if color_space == "srgb":
            ref_yuv = _rgb_to_yuv(_srgb_to_linear(ref_crop), matrix=yuv_matrix)
            img_yuv = _rgb_to_yuv(_srgb_to_linear(img_crop), matrix=yuv_matrix)
        else:
            ref_yuv = _rgb_to_yuv(ref_crop, matrix=yuv_matrix)
            img_yuv = _rgb_to_yuv(img_crop, matrix=yuv_matrix)

        mask_np = mask_crop[0, 0].detach().cpu().numpy().astype(np.float32)
        mask_bool = mask_np > 1e-3
        outside = ~mask_bool
        dist_out = distance_transform_edt(outside).astype(np.float32)
        dist_in = distance_transform_edt(mask_bool).astype(np.float32)
        outer_band = outside & (dist_out > 0.0) & (dist_out <= float(max(outer_width, 1)))
        inner_band = mask_bool & (dist_in > 0.0) & (dist_in <= float(max(int(inner_width), 1)))

        if not np.any(outer_band):
            debug_items.append({"reason": "no_outer_donor_samples", "bbox": bbox})
            continue
        if not np.any(inner_band):
            debug_items.append({"reason": "no_inner_band", "bbox": bbox})
            continue

        outer_band_t = torch.from_numpy(outer_band).to(device=ref_yuv.device)
        ref_samples = ref_yuv[:, :, outer_band_t].reshape(1, 3, -1)
        img_samples = img_yuv[:, :, outer_band_t].reshape(1, 3, -1)
        lookup = _build_delta_lookup(
            img_samples,
            ref_samples,
            bins=int(bins),
            mode=correction_mode,
            lut_mode=lut_mode,
            matrix=yuv_matrix,
        )
        ch, cw = mask_bool.shape
        rows_i, cols_i = np.where(inner_band)
        r0 = max(int(rows_i.min()) - margin, 0)
        r1 = min(int(rows_i.max()) + margin + 1, ch)
        c0 = max(int(cols_i.min()) - margin, 0)
        c1 = min(int(cols_i.max()) + margin + 1, cw)
        gen_sub = img_yuv[:, :, r0:r1, c0:c1]
        delta_sub = _lookup_delta(gen_sub, lookup) * channel_scale
        delta_sub = _gaussian_blur_band(delta_sub, float(delta_smoothing_sigma))
        delta_full = torch.zeros(1, img_yuv.shape[1], ch, cw, device=img_yuv.device, dtype=img_yuv.dtype)
        delta_full[:, :, r0:r1, c0:c1] = delta_sub

        inner_weight_np = _freeform_inner_weight(
            dist_in,
            inner_width=int(inner_width),
            flat_top_px=int(inner_flat_top_px),
        ) * inner_band.astype(np.float32)
        inner_weight = torch.from_numpy(inner_weight_np).to(device=delta_full.device, dtype=delta_full.dtype).view(1, 1, ch, cw)
        corrected_yuv = img_yuv + delta_full * inner_weight * mask_crop

        neutral = corrected_yuv[:, :1].clamp(0.0, 1.0).expand(-1, 3, -1, -1)
        if color_space == "srgb":
            corrected_rgb = _linear_to_srgb(
                _compress_to_unit_gamut(_yuv_to_rgb(corrected_yuv, matrix=yuv_matrix), neutral)
            ).clamp(0.0, 1.0)
        else:
            corrected_rgb = _compress_to_unit_gamut(
                _yuv_to_rgb(corrected_yuv, matrix=yuv_matrix), neutral
            )
        corrected_rgb = corrected_rgb * mask_crop + img_crop * (1.0 - mask_crop)
        corrected[idx : idx + 1, :, cy0:cy1, cx0:cx1] = corrected_rgb.to(dtype=image_dtype)
        debug_items.append(
            {
                "reason": "applied",
                "bbox": bbox,
                "crop": (cx0, cy0, cx1, cy1),
                "outer_samples": int(outer_band.sum()),
                "inner_pixels": int(inner_band.sum()),
            }
        )

    return corrected, {
        "reason": "applied" if any(item.get("reason") == "applied" for item in debug_items) else debug_items[0]["reason"],
        "bbox": bbox,
        "per_sample": debug_items,
        "soft_mask": soft_mask,
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
    if "per_sample" in debug:
        summary = {
            "mode": "freeform",
            "reason": debug.get("reason", "applied"),
            "bbox": list(debug.get("bbox", [])),
            "per_sample": debug.get("per_sample", []),
            "debug_root": str(root),
        }
    else:
        summary = {
            "mode": "rectangular",
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

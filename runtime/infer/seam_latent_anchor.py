from __future__ import annotations

import torch
import torch.nn.functional as F

from ..strip_ops import mask_bbox
from .merge_bands import build_seam_local_weight_map


Side = str


def _match_batch(x: torch.Tensor, batch: int) -> torch.Tensor:
    if x.shape[0] == batch:
        return x
    if x.shape[0] == 1:
        return x.expand(batch, -1, -1, -1)
    return x[:batch]


def _match_channels(x: torch.Tensor, channels: int) -> torch.Tensor:
    if x.shape[1] == channels:
        return x
    if x.shape[1] > channels:
        return x[:, :channels]
    return F.pad(x, (0, 0, 0, 0, 0, channels - x.shape[1]))


def _reduce_strip(strip: torch.Tensor, side: Side, reduce: str) -> torch.Tensor:
    if side in {"left", "right"}:
        dim = -1
    else:
        dim = -2
    if reduce == "median":
        return torch.median(strip, dim=dim, keepdim=True).values
    return torch.mean(strip, dim=dim, keepdim=True)


def _smooth_profile(profile: torch.Tensor, side: Side, kernel_size: int = 5) -> torch.Tensor:
    kernel_size = max(1, int(kernel_size))
    if kernel_size <= 1:
        return profile
    if side in {"left", "right"}:
        length = profile.shape[-2]
        if length < 3:
            return profile
        kernel_size = min(kernel_size, length if length % 2 == 1 else length - 1)
        if kernel_size <= 1:
            return profile
        pad = kernel_size // 2
        x = profile.squeeze(-1).reshape(-1, 1, length)
        x = F.avg_pool1d(F.pad(x, (pad, pad), mode="replicate"), kernel_size=kernel_size, stride=1)
        return x.reshape(profile.shape[0], profile.shape[1], length, 1)
    length = profile.shape[-1]
    if length < 3:
        return profile
    kernel_size = min(kernel_size, length if length % 2 == 1 else length - 1)
    if kernel_size <= 1:
        return profile
    pad = kernel_size // 2
    x = profile.squeeze(-2).reshape(-1, 1, length)
    x = F.avg_pool1d(F.pad(x, (pad, pad), mode="replicate"), kernel_size=kernel_size, stride=1)
    return x.reshape(profile.shape[0], profile.shape[1], 1, length)


def _extract_outer_profile(
    anchor_latent: torch.Tensor,
    bbox: tuple[int, int, int, int],
    side: Side,
    outer_width: int,
    reduce: str,
) -> torch.Tensor | None:
    _, _, height, width = anchor_latent.shape
    x0, y0, x1, y1 = bbox
    if side == "left":
        width_px = min(int(outer_width), x0)
        if width_px <= 0:
            return None
        strip = anchor_latent[:, :, y0:y1, x0 - width_px : x0]
    elif side == "right":
        width_px = min(int(outer_width), width - x1)
        if width_px <= 0:
            return None
        strip = anchor_latent[:, :, y0:y1, x1 : x1 + width_px]
    elif side == "top":
        width_px = min(int(outer_width), y0)
        if width_px <= 0:
            return None
        strip = anchor_latent[:, :, y0 - width_px : y0, x0:x1]
    elif side == "bottom":
        width_px = min(int(outer_width), height - y1)
        if width_px <= 0:
            return None
        strip = anchor_latent[:, :, y1 : y1 + width_px, x0:x1]
    else:
        raise ValueError(f"unsupported side: {side}")
    return _reduce_strip(strip, side, reduce)


def _extract_inner_profile(
    denoised: torch.Tensor,
    bbox: tuple[int, int, int, int],
    side: Side,
    sample_width: int,
    reduce: str,
) -> torch.Tensor | None:
    _, _, height, width = denoised.shape
    x0, y0, x1, y1 = bbox
    if side == "left":
        band = min(int(sample_width), max(x1 - x0, 0))
        if band <= 0:
            return None
        strip = denoised[:, :, y0:y1, x0 : x0 + band]
    elif side == "right":
        band = min(int(sample_width), max(x1 - x0, 0))
        if band <= 0:
            return None
        strip = denoised[:, :, y0:y1, x1 - band : x1]
    elif side == "top":
        band = min(int(sample_width), max(y1 - y0, 0))
        if band <= 0:
            return None
        strip = denoised[:, :, y0 : y0 + band, x0:x1]
    elif side == "bottom":
        band = min(int(sample_width), max(y1 - y0, 0))
        if band <= 0:
            return None
        strip = denoised[:, :, y1 - band : y1, x0:x1]
    else:
        raise ValueError(f"unsupported side: {side}")
    return _reduce_strip(strip, side, reduce)


def _reduce_strip_std(strip: torch.Tensor, side: Side) -> torch.Tensor:
    if side in {"left", "right"}:
        dim = -1
    else:
        dim = -2
    return torch.std(strip, dim=dim, keepdim=True, unbiased=False)


def _extract_outer_std(
    anchor_latent: torch.Tensor,
    bbox: tuple[int, int, int, int],
    side: Side,
    outer_width: int,
) -> torch.Tensor | None:
    _, _, height, width = anchor_latent.shape
    x0, y0, x1, y1 = bbox
    if side == "left":
        width_px = min(int(outer_width), x0)
        if width_px <= 0:
            return None
        strip = anchor_latent[:, :, y0:y1, x0 - width_px : x0]
    elif side == "right":
        width_px = min(int(outer_width), width - x1)
        if width_px <= 0:
            return None
        strip = anchor_latent[:, :, y0:y1, x1 : x1 + width_px]
    elif side == "top":
        width_px = min(int(outer_width), y0)
        if width_px <= 0:
            return None
        strip = anchor_latent[:, :, y0 - width_px : y0, x0:x1]
    elif side == "bottom":
        width_px = min(int(outer_width), height - y1)
        if width_px <= 0:
            return None
        strip = anchor_latent[:, :, y1 : y1 + width_px, x0:x1]
    else:
        raise ValueError(f"unsupported side: {side}")
    return _reduce_strip_std(strip, side)


def _extract_inner_std(
    x: torch.Tensor,
    bbox: tuple[int, int, int, int],
    side: Side,
    sample_width: int,
) -> torch.Tensor | None:
    _, _, height, width = x.shape
    x0, y0, x1, y1 = bbox
    if side == "left":
        band = min(int(sample_width), max(x1 - x0, 0))
        if band <= 0:
            return None
        strip = x[:, :, y0:y1, x0 : x0 + band]
    elif side == "right":
        band = min(int(sample_width), max(x1 - x0, 0))
        if band <= 0:
            return None
        strip = x[:, :, y0:y1, x1 - band : x1]
    elif side == "top":
        band = min(int(sample_width), max(y1 - y0, 0))
        if band <= 0:
            return None
        strip = x[:, :, y0 : y0 + band, x0:x1]
    elif side == "bottom":
        band = min(int(sample_width), max(y1 - y0, 0))
        if band <= 0:
            return None
        strip = x[:, :, y1 - band : y1, x0:x1]
    else:
        raise ValueError(f"unsupported side: {side}")
    return _reduce_strip_std(strip, side)


def _extract_inner_band(
    x: torch.Tensor,
    bbox: tuple[int, int, int, int],
    side: Side,
    band_size: int,
) -> torch.Tensor | None:
    _, _, height, width = x.shape
    x0, y0, x1, y1 = bbox
    if side == "left":
        if band_size <= 0:
            return None
        return x[:, :, y0:y1, x0 : x0 + band_size]
    if side == "right":
        if band_size <= 0:
            return None
        return x[:, :, y0:y1, x1 - band_size : x1]
    if side == "top":
        if band_size <= 0:
            return None
        return x[:, :, y0 : y0 + band_size, x0:x1]
    if side == "bottom":
        if band_size <= 0:
            return None
        return x[:, :, y1 - band_size : y1, x0:x1]
    raise ValueError(f"unsupported side: {side}")


def _place_band(
    band: torch.Tensor,
    bbox: tuple[int, int, int, int],
    side: Side,
    full_shape: tuple[int, int, int, int],
) -> torch.Tensor:
    batch, channels, height, width = full_shape
    target = torch.zeros(batch, channels, height, width, device=band.device, dtype=band.dtype)
    x0, y0, x1, y1 = bbox
    if side == "left":
        target[:, :, y0:y1, x0 : x0 + band.shape[-1]] = band
    elif side == "right":
        target[:, :, y0:y1, x1 - band.shape[-1] : x1] = band
    elif side == "top":
        target[:, :, y0 : y0 + band.shape[-2], x0:x1] = band
    elif side == "bottom":
        target[:, :, y1 - band.shape[-2] : y1, x0:x1] = band
    else:
        raise ValueError(f"unsupported side: {side}")
    return target


def _place_profile(
    profile: torch.Tensor,
    bbox: tuple[int, int, int, int],
    side: Side,
    band_size: int,
    full_shape: tuple[int, int, int, int],
) -> torch.Tensor:
    batch, channels, height, width = full_shape
    target = torch.zeros(batch, channels, height, width, device=profile.device, dtype=profile.dtype)
    x0, y0, x1, y1 = bbox
    if side == "left":
        target[:, :, y0:y1, x0 : x0 + band_size] = profile.expand(-1, -1, -1, band_size)
    elif side == "right":
        target[:, :, y0:y1, x1 - band_size : x1] = profile.expand(-1, -1, -1, band_size)
    elif side == "top":
        target[:, :, y0 : y0 + band_size, x0:x1] = profile.expand(-1, -1, band_size, -1)
    elif side == "bottom":
        target[:, :, y1 - band_size : y1, x0:x1] = profile.expand(-1, -1, band_size, -1)
    else:
        raise ValueError(f"unsupported side: {side}")
    return target


def prepare_seam_anchor_state(
    anchor_latent: torch.Tensor,
    mask: torch.Tensor,
    *,
    anchor_width_px: int,
    anchor_falloff_px: int,
    process_left: bool = True,
    process_right: bool = True,
    process_top: bool = True,
    process_bottom: bool = True,
    reduce: str = "mean",
    profile_smooth_kernel: int = 5,
) -> dict:
    if anchor_latent.ndim != 4:
        raise ValueError("expected anchor_latent with shape [B,C,H,W]")
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(1)
    elif mask.ndim != 4:
        raise ValueError("expected mask with shape [H,W], [B,H,W], or [B,1,H,W]")

    mask = mask.float()
    if mask.shape[-2:] != anchor_latent.shape[-2:]:
        mask = F.interpolate(mask, size=anchor_latent.shape[-2:], mode="nearest")
    mask = (mask > 0.5).float()
    bbox = mask_bbox(mask)
    x0, y0, x1, y1 = bbox

    sides: list[Side] = []
    if process_left and x0 > 0:
        sides.append("left")
    if process_right and x1 < anchor_latent.shape[-1]:
        sides.append("right")
    if process_top and y0 > 0:
        sides.append("top")
    if process_bottom and y1 < anchor_latent.shape[-2]:
        sides.append("bottom")

    per_side_profiles: dict[Side, torch.Tensor] = {}
    per_side_weights: dict[Side, torch.Tensor] = {}
    per_side_band_sizes: dict[Side, int] = {}
    per_side_sample_widths: dict[Side, int] = {}
    per_side_std_profiles: dict[Side, torch.Tensor] = {}

    for side in sides:
        profile = _extract_outer_profile(anchor_latent, bbox, side, anchor_width_px, reduce)
        std_profile = _extract_outer_std(anchor_latent, bbox, side, anchor_width_px)
        if profile is None:
            continue
        profile = _smooth_profile(profile, side, kernel_size=profile_smooth_kernel)
        if std_profile is None:
            continue
        std_profile = _smooth_profile(std_profile, side, kernel_size=profile_smooth_kernel)
        if side in {"left", "right"}:
            band_size = min(int(anchor_falloff_px), max(x1 - x0, 0))
            sample_width = min(int(anchor_width_px), max(x1 - x0, 0))
        else:
            band_size = min(int(anchor_falloff_px), max(y1 - y0, 0))
            sample_width = min(int(anchor_width_px), max(y1 - y0, 0))
        if band_size <= 0 or sample_width <= 0:
            continue
        weight = build_seam_local_weight_map(
            mask,
            bbox,
            side,
            band_size,
            blend_falloff_px=0,
            power=1.0,
        )
        if float(weight.max().item()) <= 0.0:
            continue
        per_side_profiles[side] = profile
        per_side_weights[side] = weight
        per_side_band_sizes[side] = int(band_size)
        per_side_sample_widths[side] = int(sample_width)
        per_side_std_profiles[side] = std_profile

    return {
        "bbox": bbox,
        "mask": mask,
        "sides": tuple(per_side_profiles.keys()),
        "profiles": per_side_profiles,
        "std_profiles": per_side_std_profiles,
        "weights": per_side_weights,
        "band_sizes": per_side_band_sizes,
        "sample_widths": per_side_sample_widths,
        "reduce": reduce,
    }


def apply_seam_anchor_correction(
    denoised: torch.Tensor,
    anchor_state: dict,
    effective_strength: float,
) -> torch.Tensor:
    if effective_strength <= 0.0 or not anchor_state.get("sides"):
        return denoised

    bbox = anchor_state["bbox"]
    reduce = anchor_state["reduce"]
    profiles = anchor_state["profiles"]
    weights = anchor_state["weights"]
    band_sizes = anchor_state["band_sizes"]
    sample_widths = anchor_state["sample_widths"]

    zero = torch.zeros_like(denoised)
    weighted_sum = zero
    total_weight = torch.zeros(denoised.shape[0], 1, denoised.shape[-2], denoised.shape[-1], device=denoised.device, dtype=denoised.dtype)
    for side in anchor_state["sides"]:
        anchor_profile = _match_batch(profiles[side], denoised.shape[0]).to(device=denoised.device, dtype=denoised.dtype)
        anchor_profile = _match_channels(anchor_profile, denoised.shape[1])
        current_profile = _extract_inner_profile(denoised, bbox, side, sample_widths[side], reduce)
        if current_profile is None:
            continue
        current_profile = _match_batch(current_profile, denoised.shape[0]).to(device=denoised.device, dtype=denoised.dtype)
        current_profile = _match_channels(current_profile, denoised.shape[1])
        correction_profile = anchor_profile - current_profile
        correction_map = _place_profile(correction_profile, bbox, side, band_sizes[side], denoised.shape)
        weight = _match_batch(weights[side], denoised.shape[0]).to(device=denoised.device, dtype=denoised.dtype)
        if weight.shape[-2:] != denoised.shape[-2:]:
            weight = F.interpolate(weight, size=denoised.shape[-2:], mode="bilinear", align_corners=False)
        weighted_sum = weighted_sum + correction_map * weight
        total_weight = total_weight + weight

    if float(total_weight.max().item()) <= 0.0:
        return denoised
    coverage = total_weight.clamp(0.0, 1.0)
    merged = torch.where(total_weight > 1.0, weighted_sum / total_weight.clamp_min(1e-8), weighted_sum)
    merged = torch.where(total_weight > 1.0, merged * coverage, merged)
    return denoised + merged * float(effective_strength)


def apply_seam_latent_guidance(
    x: torch.Tensor,
    anchor_state: dict,
    effective_strength: float,
    *,
    mode: str = "mean_shift",
    boundary_only: bool = False,
    variance_limit: float = 2.0,
) -> torch.Tensor:
    if effective_strength <= 0.0 or not anchor_state.get("sides"):
        return x

    bbox = anchor_state["bbox"]
    reduce = anchor_state["reduce"]
    profiles = anchor_state["profiles"]
    std_profiles = anchor_state["std_profiles"]
    weights = anchor_state["weights"]
    band_sizes = anchor_state["band_sizes"]
    sample_widths = anchor_state["sample_widths"]

    weighted_sum = torch.zeros_like(x)
    total_weight = torch.zeros(x.shape[0], 1, x.shape[-2], x.shape[-1], device=x.device, dtype=x.dtype)
    for side in anchor_state["sides"]:
        weight = _match_batch(weights[side], x.shape[0]).to(device=x.device, dtype=x.dtype)
        if boundary_only:
            # Sharpen the falloff so direct guidance stays close to the seam edge.
            weight = weight.clamp(0.0, 1.0).pow(2.0)
        anchor_profile = _match_batch(profiles[side], x.shape[0]).to(device=x.device, dtype=x.dtype)
        anchor_profile = _match_channels(anchor_profile, x.shape[1])
        current_profile = _extract_inner_profile(x, bbox, side, sample_widths[side], reduce)
        if current_profile is None:
            continue
        current_profile = _match_batch(current_profile, x.shape[0]).to(device=x.device, dtype=x.dtype)
        current_profile = _match_channels(current_profile, x.shape[1])
        if mode == "matched_noise":
            inner_band = _extract_inner_band(x, bbox, side, band_sizes[side])
            if inner_band is None:
                continue
            current_std = _extract_inner_std(x, bbox, side, sample_widths[side])
            anchor_std = _match_batch(std_profiles[side], x.shape[0]).to(device=x.device, dtype=x.dtype)
            anchor_std = _match_channels(anchor_std, x.shape[1])
            if current_std is None:
                continue
            current_std = _match_batch(current_std, x.shape[0]).to(device=x.device, dtype=x.dtype)
            current_std = _match_channels(current_std, x.shape[1])
            scale = (anchor_std / current_std.clamp_min(1e-4)).clamp(1.0 / variance_limit, variance_limit)
            if side in {"left", "right"}:
                mean_exp = anchor_profile.expand(-1, -1, -1, band_sizes[side])
                cur_mean_exp = current_profile.expand(-1, -1, -1, band_sizes[side])
                scale_exp = scale.expand(-1, -1, -1, band_sizes[side])
            else:
                mean_exp = anchor_profile.expand(-1, -1, band_sizes[side], -1)
                cur_mean_exp = current_profile.expand(-1, -1, band_sizes[side], -1)
                scale_exp = scale.expand(-1, -1, band_sizes[side], -1)
            matched_band = (inner_band - cur_mean_exp) * scale_exp + mean_exp
            delta_map = _place_band(matched_band - inner_band, bbox, side, x.shape)
        else:
            correction_profile = anchor_profile - current_profile
            delta_map = _place_profile(correction_profile, bbox, side, band_sizes[side], x.shape)
        if weight.shape[-2:] != x.shape[-2:]:
            weight = F.interpolate(weight, size=x.shape[-2:], mode="bilinear", align_corners=False)
        weighted_sum = weighted_sum + delta_map * weight
        total_weight = total_weight + weight

    if float(total_weight.max().item()) <= 0.0:
        return x
    coverage = total_weight.clamp(0.0, 1.0)
    merged = torch.where(total_weight > 1.0, weighted_sum / total_weight.clamp_min(1e-8), weighted_sum)
    merged = torch.where(total_weight > 1.0, merged * coverage, merged)
    return x + merged * float(effective_strength)

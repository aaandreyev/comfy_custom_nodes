from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..strip_ops import mask_bbox
from .merge_bands import build_seam_local_weight_map


Side = str


@dataclass(frozen=True)
class SeamAnchorDebug:
    bbox: tuple[int, int, int, int]
    sides: tuple[Side, ...]


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


def _build_side_target_map(
    anchor_latent: torch.Tensor,
    bbox: tuple[int, int, int, int],
    side: Side,
    anchor_width_px: int,
    anchor_falloff_px: int,
    reduce: str,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    batch, channels, height, width = anchor_latent.shape
    x0, y0, x1, y1 = bbox
    band_w = max(0, min(int(anchor_falloff_px), x1 - x0))
    band_h = max(0, min(int(anchor_falloff_px), y1 - y0))
    if side == "left":
        outer_w = min(int(anchor_width_px), x0)
        if outer_w <= 0 or band_w <= 0:
            return None, None
        strip = anchor_latent[:, :, y0:y1, x0 - outer_w : x0]
        profile = _reduce_strip(strip, side, reduce)
        target = torch.zeros(batch, channels, height, width, device=anchor_latent.device, dtype=anchor_latent.dtype)
        target[:, :, y0:y1, x0 : x0 + band_w] = profile.expand(-1, -1, -1, band_w)
        return target, profile
    if side == "right":
        outer_w = min(int(anchor_width_px), width - x1)
        if outer_w <= 0 or band_w <= 0:
            return None, None
        strip = anchor_latent[:, :, y0:y1, x1 : x1 + outer_w]
        profile = _reduce_strip(strip, side, reduce)
        target = torch.zeros(batch, channels, height, width, device=anchor_latent.device, dtype=anchor_latent.dtype)
        target[:, :, y0:y1, x1 - band_w : x1] = profile.expand(-1, -1, -1, band_w)
        return target, profile
    if side == "top":
        outer_h = min(int(anchor_width_px), y0)
        if outer_h <= 0 or band_h <= 0:
            return None, None
        strip = anchor_latent[:, :, y0 - outer_h : y0, x0:x1]
        profile = _reduce_strip(strip, side, reduce)
        target = torch.zeros(batch, channels, height, width, device=anchor_latent.device, dtype=anchor_latent.dtype)
        target[:, :, y0 : y0 + band_h, x0:x1] = profile.expand(-1, -1, band_h, -1)
        return target, profile
    if side == "bottom":
        outer_h = min(int(anchor_width_px), height - y1)
        if outer_h <= 0 or band_h <= 0:
            return None, None
        strip = anchor_latent[:, :, y1 : y1 + outer_h, x0:x1]
        profile = _reduce_strip(strip, side, reduce)
        target = torch.zeros(batch, channels, height, width, device=anchor_latent.device, dtype=anchor_latent.dtype)
        target[:, :, y1 - band_h : y1, x0:x1] = profile.expand(-1, -1, band_h, -1)
        return target, profile
    raise ValueError(f"unsupported side: {side}")


def prepare_seam_anchor_maps(
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
) -> tuple[torch.Tensor, torch.Tensor, dict]:
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

    batch, channels, height, width = anchor_latent.shape
    zero_target = torch.zeros(batch, channels, height, width, device=anchor_latent.device, dtype=anchor_latent.dtype)
    zero_weight = torch.zeros(batch, 1, height, width, device=anchor_latent.device, dtype=anchor_latent.dtype)
    if not sides or anchor_width_px <= 0 or anchor_falloff_px <= 0:
        return zero_target, zero_weight, {"bbox": bbox, "sides": tuple(), "profiles": {}, "weights": {}}

    per_side_targets: dict[Side, torch.Tensor] = {}
    per_side_weights: dict[Side, torch.Tensor] = {}
    per_side_profiles: dict[Side, torch.Tensor] = {}
    for side in sides:
        target, profile = _build_side_target_map(
            anchor_latent,
            bbox,
            side,
            anchor_width_px,
            anchor_falloff_px,
            reduce,
        )
        if target is None or profile is None:
            continue
        weight = build_seam_local_weight_map(
            mask,
            bbox,
            side,
            int(anchor_falloff_px),
            blend_falloff_px=0,
            power=1.0,
        )
        support = (target.abs().mean(dim=1, keepdim=True) > 1e-8).to(anchor_latent.dtype)
        weight = weight * support
        if float(weight.max().item()) <= 0.0:
            continue
        per_side_targets[side] = target
        per_side_weights[side] = weight
        per_side_profiles[side] = profile

    if not per_side_targets:
        return zero_target, zero_weight, {"bbox": bbox, "sides": tuple(), "profiles": {}, "weights": {}}

    target_stack = torch.stack([per_side_targets[side] for side in per_side_targets], dim=0)
    weight_stack = torch.stack([per_side_weights[side] for side in per_side_targets], dim=0)
    weighted_target = (target_stack * weight_stack).sum(dim=0)
    total_weight = weight_stack.sum(dim=0)
    merged_target = torch.where(total_weight > 1e-8, weighted_target / total_weight.clamp_min(1e-8), zero_target)
    merged_weight = total_weight.clamp(0.0, 1.0)
    debug = {
        "bbox": bbox,
        "sides": tuple(per_side_targets.keys()),
        "profiles": per_side_profiles,
        "weights": per_side_weights,
    }
    return merged_target * mask, merged_weight * mask, debug


def apply_seam_anchor_correction(
    denoised: torch.Tensor,
    target_map: torch.Tensor,
    weight_map: torch.Tensor,
    effective_strength: float,
) -> torch.Tensor:
    target = _match_batch(target_map, denoised.shape[0]).to(device=denoised.device, dtype=denoised.dtype)
    target = _match_channels(target, denoised.shape[1])
    weight = _match_batch(weight_map, denoised.shape[0]).to(device=denoised.device, dtype=denoised.dtype)
    if target.shape[-2:] != denoised.shape[-2:]:
        target = F.interpolate(target, size=denoised.shape[-2:], mode="bilinear", align_corners=False)
    if weight.shape[-2:] != denoised.shape[-2:]:
        weight = F.interpolate(weight, size=denoised.shape[-2:], mode="bilinear", align_corners=False)
    effective = weight.clamp(0.0, 1.0) * float(effective_strength)
    return denoised + (target - denoised) * effective

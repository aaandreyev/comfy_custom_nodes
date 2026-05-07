from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ..strip_ops import mask_bbox
from .merge_bands import build_seam_local_weight_map


Position = str

CORNER_POSITIONS: tuple[Position, ...] = ("nw", "ne", "sw", "se")
SIDE_POSITIONS: tuple[Position, ...] = ("n", "s", "w", "e")
ALL_POSITIONS: tuple[Position, ...] = ("nw", "n", "ne", "w", "e", "sw", "s", "se")


def normalize_mask(mask: torch.Tensor, reference_shape: tuple[int, int]) -> torch.Tensor:
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(1)
    elif mask.ndim != 4:
        raise ValueError("expected mask with shape [H,W], [B,H,W], or [B,1,H,W]")
    mask = mask.float()
    if mask.shape[-2:] != reference_shape:
        mask = F.interpolate(mask, size=reference_shape, mode="nearest")
    return mask.clamp(0.0, 1.0)


def position_zone_mask(
    shape: tuple[int, int],
    bbox: tuple[int, int, int, int],
    position: Position,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    height, width = shape
    x0, y0, x1, y1 = bbox
    out = torch.zeros(1, 1, height, width, device=device, dtype=dtype)
    if position == "nw":
        out[:, :, :y0, :x0] = 1.0
    elif position == "n":
        out[:, :, :y0, x0:x1] = 1.0
    elif position == "ne":
        out[:, :, :y0, x1:] = 1.0
    elif position == "w":
        out[:, :, y0:y1, :x0] = 1.0
    elif position == "e":
        out[:, :, y0:y1, x1:] = 1.0
    elif position == "sw":
        out[:, :, y1:, :x0] = 1.0
    elif position == "s":
        out[:, :, y1:, x0:x1] = 1.0
    elif position == "se":
        out[:, :, y1:, x1:] = 1.0
    else:
        raise ValueError(f"unsupported position: {position}")
    return out


def parse_present_positions(
    topology_mask: torch.Tensor | None,
    bbox: tuple[int, int, int, int],
    shape: tuple[int, int],
    *,
    threshold: float,
    device: torch.device,
    dtype: torch.dtype,
) -> set[Position]:
    if topology_mask is None:
        return set()
    norm = normalize_mask(topology_mask, shape).to(device=device, dtype=dtype)
    present: set[Position] = set()
    for position in ALL_POSITIONS:
        zone = position_zone_mask(shape, bbox, position, device=device, dtype=dtype)
        denom = float(zone.sum().item())
        if denom <= 0.0:
            continue
        coverage = float((norm * zone).sum().item() / denom)
        if coverage > float(threshold):
            present.add(position)
    return present


def _build_band_alpha_from_distance(
    distance: torch.Tensor,
    *,
    band_width: int,
    band_gradient_px: int,
) -> torch.Tensor:
    flat_top_px = max(int(band_width) - max(int(band_gradient_px), 0), 0)
    width = max(float(band_width), 1.0)
    fade_start = min(max(float(flat_top_px), 0.0), max(width - 1.0, 0.0))
    alpha = torch.where(
        distance <= fade_start,
        torch.ones_like(distance),
        1.0 - ((distance - fade_start) / max(width - fade_start, 1.0)).clamp(0.0, 1.0),
    )
    return 0.5 * (1.0 - torch.cos(math.pi * alpha.clamp(0.0, 1.0)))


def _corner_weight(
    mask: torch.Tensor,
    bbox: tuple[int, int, int, int],
    corner: Position,
    *,
    band_width: int,
    band_gradient_px: int,
) -> torch.Tensor:
    _, _, height, width = mask.shape
    x0, y0, x1, y1 = [int(x) for x in bbox]
    device, dtype = mask.device, mask.dtype
    yy = torch.arange(height, device=device, dtype=dtype).view(1, 1, height, 1)
    xx = torch.arange(width, device=device, dtype=dtype).view(1, 1, 1, width)

    if corner == "nw":
        dx = (xx - float(x0)).clamp_min(0.0)
        dy = (yy - float(y0)).clamp_min(0.0)
    elif corner == "ne":
        dx = (float(x1 - 1) - xx).clamp_min(0.0)
        dy = (yy - float(y0)).clamp_min(0.0)
    elif corner == "sw":
        dx = (xx - float(x0)).clamp_min(0.0)
        dy = (float(y1 - 1) - yy).clamp_min(0.0)
    elif corner == "se":
        dx = (float(x1 - 1) - xx).clamp_min(0.0)
        dy = (float(y1 - 1) - yy).clamp_min(0.0)
    else:
        raise ValueError(f"unsupported corner: {corner}")

    distance = torch.maximum(dx, dy)
    return _build_band_alpha_from_distance(
        distance,
        band_width=int(band_width),
        band_gradient_px=int(band_gradient_px),
    ) * mask


def _zone_weight(
    mask: torch.Tensor,
    bbox: tuple[int, int, int, int],
    position: Position,
    *,
    band_width: int,
    band_gradient_px: int,
    active_sides: set[str],
) -> torch.Tensor:
    if position == "n":
        return build_seam_local_weight_map(
            mask,
            bbox,
            "top",
            int(band_width),
            flat_top_px=max(int(band_width) - max(int(band_gradient_px), 0), 0),
            blend_falloff_px=int(band_gradient_px),
            power=1.0,
            active_sides=active_sides,
        )
    if position == "s":
        return build_seam_local_weight_map(
            mask,
            bbox,
            "bottom",
            int(band_width),
            flat_top_px=max(int(band_width) - max(int(band_gradient_px), 0), 0),
            blend_falloff_px=int(band_gradient_px),
            power=1.0,
            active_sides=active_sides,
        )
    if position == "w":
        return build_seam_local_weight_map(
            mask,
            bbox,
            "left",
            int(band_width),
            flat_top_px=max(int(band_width) - max(int(band_gradient_px), 0), 0),
            blend_falloff_px=int(band_gradient_px),
            power=1.0,
            active_sides=active_sides,
        )
    if position == "e":
        return build_seam_local_weight_map(
            mask,
            bbox,
            "right",
            int(band_width),
            flat_top_px=max(int(band_width) - max(int(band_gradient_px), 0), 0),
            blend_falloff_px=int(band_gradient_px),
            power=1.0,
            active_sides=active_sides,
        )
    return _corner_weight(
        mask,
        bbox,
        position,
        band_width=int(band_width),
        band_gradient_px=int(band_gradient_px),
    )


def build_local_denoise_state(
    mask: torch.Tensor,
    *,
    global_denoise: float,
    topology_mask: torch.Tensor | None,
    local_denoise_enabled: bool,
    band_width: int,
    band_gradient_px: int,
    band_denoise_min: float,
    band_denoise_max: float,
    merge_mode: str,
    topology_threshold: float,
    preserve_outside_latent: bool,
) -> dict:
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError("mask must be [B,1,H,W]")
    if band_denoise_min > band_denoise_max:
        raise ValueError("band_denoise_min must be less than or equal to band_denoise_max")
    if topology_mask is not None:
        if topology_mask.ndim != 4 or topology_mask.shape[1] != 1:
            raise ValueError("topology_mask must be [B,1,H,W]")
        if topology_mask.shape[0] not in (1, mask.shape[0]):
            raise ValueError("topology_mask batch must be 1 or match mask batch size")

    per_sample = []
    maps = []
    bbox_list = []
    present_union: set[Position] = set()
    global_denoise = float(max(0.0, min(1.0, global_denoise)))
    band_denoise_min = float(max(0.0, min(1.0, band_denoise_min)))
    band_denoise_max = float(max(0.0, min(1.0, band_denoise_max)))
    normalized_topology = None
    if topology_mask is not None:
        normalized_topology = normalize_mask(topology_mask, mask.shape[-2:]).to(device=mask.device, dtype=mask.dtype)

    for idx in range(mask.shape[0]):
        sample_mask = mask[idx : idx + 1]
        bbox = mask_bbox(sample_mask)
        bbox_list.append(bbox)
        x0, y0, x1, y1 = bbox
        bbox_w = max(x1 - x0, 1)
        bbox_h = max(y1 - y0, 1)
        present_positions = parse_present_positions(
            normalized_topology[idx : idx + 1] if normalized_topology is not None and normalized_topology.shape[0] > 1 else normalized_topology,
            bbox,
            sample_mask.shape[-2:],
            threshold=float(topology_threshold),
            device=sample_mask.device,
            dtype=sample_mask.dtype,
        )
        present_union.update(present_positions)

        if preserve_outside_latent:
            base_map = torch.zeros_like(sample_mask)
        else:
            base_map = torch.full_like(sample_mask, global_denoise)
        base_map = torch.where(sample_mask > 0.5, torch.full_like(sample_mask, global_denoise), base_map)

        if not local_denoise_enabled:
            target_map = base_map.clamp(0.0, 1.0)
            maps.append(target_map)
            per_sample.append({"bbox": bbox, "present_positions": tuple(sorted(present_positions))})
            continue

        active_sides = {
            side
            for side, pos in (("left", "w"), ("right", "e"), ("top", "n"), ("bottom", "s"))
            if pos in present_positions
        }
        effective_positions = set(present_positions)
        if not effective_positions:
            effective_positions = set(ALL_POSITIONS)
            active_sides = {"left", "right", "top", "bottom"}

        if int(band_width) >= max(bbox_w, bbox_h):
            target_map = torch.where(
                sample_mask > 0.5,
                torch.full_like(sample_mask, band_denoise_max),
                base_map,
            ).clamp(0.0, 1.0)
            maps.append(target_map)
            per_sample.append(
                {
                    "bbox": bbox,
                    "present_positions": tuple(sorted(present_positions)),
                    "uniform_fallback": True,
                }
            )
            continue

        zone_weights = []
        zone_values = []
        for position in sorted(effective_positions):
            weight = _zone_weight(
                sample_mask,
                bbox,
                position,
                band_width=int(band_width),
                band_gradient_px=int(band_gradient_px),
                active_sides=active_sides,
            ).clamp(0.0, 1.0)
            if float(weight.max().item()) <= 0.0:
                continue
            value = band_denoise_min + (band_denoise_max - band_denoise_min) * weight
            zone_weights.append(weight)
            zone_values.append(value)

        if not zone_weights:
            target_map = base_map.clamp(0.0, 1.0)
        else:
            alpha_stack = torch.stack(zone_weights, dim=0)
            value_stack = torch.stack(zone_values, dim=0)
            alpha = alpha_stack.max(dim=0).values
            if merge_mode == "max":
                gather_idx = alpha_stack.argmax(dim=0, keepdim=True)
                local_value = torch.gather(value_stack, 0, gather_idx).squeeze(0)
            elif merge_mode == "normalized_sum":
                local_value = (value_stack * alpha_stack).sum(dim=0) / alpha_stack.sum(dim=0).clamp_min(1e-8)
            else:
                raise ValueError(f"unsupported merge_mode: {merge_mode}")
            target_map = torch.where(
                alpha > 0.0,
                base_map * (1.0 - alpha) + local_value * alpha,
                base_map,
            ).clamp(0.0, 1.0)

        maps.append(target_map)
        per_sample.append(
            {
                "bbox": bbox,
                "present_positions": tuple(sorted(present_positions)),
                "uniform_fallback": False,
            }
        )

    target_denoise_map = torch.cat(maps, dim=0)
    return {
        "target_denoise_map": target_denoise_map,
        "per_sample": per_sample,
        "bbox": bbox_list[0] if len(bbox_list) == 1 else tuple(bbox_list),
        "present_positions": tuple(sorted(present_union)),
    }


def build_continuous_schedule(full_schedule: list[float], denoise: float) -> list[float]:
    if not full_schedule:
        return []
    denoise = float(max(0.0, min(1.0, denoise)))
    first = float(full_schedule[0])
    last = float(full_schedule[-1])
    if denoise >= first - 1e-8:
        return [float(x) for x in full_schedule]
    if denoise <= last + 1e-8:
        return [denoise]
    for idx in range(len(full_schedule) - 1):
        hi = float(full_schedule[idx])
        lo = float(full_schedule[idx + 1])
        if hi >= denoise >= lo:
            if abs(hi - denoise) <= 1e-8:
                return [float(x) for x in full_schedule[idx:]]
            return [denoise] + [float(x) for x in full_schedule[idx + 1 :]]
    return [denoise, last]


def apply_spatial_denoise_preservation(
    x: torch.Tensor,
    latent: torch.Tensor,
    noise: torch.Tensor,
    target_denoise_map: torch.Tensor,
    t_next: float,
) -> torch.Tensor:
    t_next = float(t_next)
    if t_next <= 1e-6:
        return x
    target = target_denoise_map.to(device=x.device, dtype=x.dtype).clamp(0.0, 1.0)
    allowed = torch.minimum(target, torch.full_like(target, t_next))
    freedom = (allowed / max(t_next, 1e-6)).clamp(0.0, 1.0)
    ref_x = (1.0 - allowed) * latent + allowed * noise
    return x * freedom + ref_x * (1.0 - freedom)

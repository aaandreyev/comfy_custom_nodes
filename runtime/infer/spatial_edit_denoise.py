from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ..strip_ops import mask_bbox
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
    norm = topology_mask.to(device=device, dtype=dtype)
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


def _shift_mask(mask: torch.Tensor, *, dx: int, dy: int) -> torch.Tensor:
    _, _, height, width = mask.shape
    out = torch.zeros_like(mask)
    src_x0 = max(-dx, 0)
    src_x1 = min(width - dx, width)
    src_y0 = max(-dy, 0)
    src_y1 = min(height - dy, height)
    dst_x0 = max(dx, 0)
    dst_x1 = dst_x0 + max(src_x1 - src_x0, 0)
    dst_y0 = max(dy, 0)
    dst_y1 = dst_y0 + max(src_y1 - src_y0, 0)
    if src_x1 <= src_x0 or src_y1 <= src_y0:
        return out
    out[:, :, dst_y0:dst_y1, dst_x0:dst_x1] = mask[:, :, src_y0:src_y1, src_x0:src_x1]
    return out


def _zone_weight_from_bbox(
    mask: torch.Tensor,
    position: Position,
    bbox: tuple[int, int, int, int],
    *,
    band_width: int,
    band_gradient_px: int,
) -> torch.Tensor:
    """
    Vectorized O(1) distance-map weight for a uniform (binary zone) source.
    Called when there is no topology mask — the source equals the full zone.
    Eliminates the per-pixel shift loop entirely.
    """
    if band_width <= 0:
        return torch.zeros_like(mask)

    device, dtype = mask.device, mask.dtype
    _, _, H, W = mask.shape
    x0, y0, x1, y1 = bbox

    xs = torch.arange(W, device=device, dtype=dtype).view(1, 1, 1, W)
    ys = torch.arange(H, device=device, dtype=dtype).view(1, 1, H, 1)

    if position == "w":
        dist = (xs - x0).clamp(min=0)
    elif position == "e":
        dist = (x1 - 1 - xs).clamp(min=0)
    elif position == "n":
        dist = (ys - y0).clamp(min=0)
    elif position == "s":
        dist = (y1 - 1 - ys).clamp(min=0)
    elif position == "nw":
        dist = torch.maximum((xs - x0).clamp(min=0), (ys - y0).clamp(min=0))
    elif position == "ne":
        dist = torch.maximum((x1 - 1 - xs).clamp(min=0), (ys - y0).clamp(min=0))
    elif position == "sw":
        dist = torch.maximum((xs - x0).clamp(min=0), (y1 - 1 - ys).clamp(min=0))
    elif position == "se":
        dist = torch.maximum((x1 - 1 - xs).clamp(min=0), (y1 - 1 - ys).clamp(min=0))
    else:
        raise ValueError(f"unsupported position: {position}")

    alpha_map = _build_band_alpha_from_distance(dist, band_width=band_width, band_gradient_px=band_gradient_px)
    return (alpha_map * mask).clamp(0.0, 1.0)


def _zone_weight(
    mask: torch.Tensor,
    source: torch.Tensor,
    position: Position,
    *,
    band_width: int,
    band_gradient_px: int,
) -> torch.Tensor:
    """
    Projection-based weight for topology-weighted source.
    O(band_width) for sides, O(band_width²) for corners.
    Alphas are precomputed in a single vectorized call before the shift loop.
    """
    if band_width <= 0:
        return torch.zeros_like(mask)

    device, dtype = mask.device, mask.dtype
    max_distance = band_width

    # Преcompute all alphas in one vectorized call (eliminates N scalar tensor creations)
    dist_range = torch.arange(max_distance, device=device, dtype=dtype)
    alpha_vec = _build_band_alpha_from_distance(
        dist_range, band_width=band_width, band_gradient_px=band_gradient_px
    )
    alpha_list = alpha_vec.tolist()

    if position == "w":
        offsets = [((d + 1, 0), alpha_list[d]) for d in range(max_distance)]
    elif position == "e":
        offsets = [((-(d + 1), 0), alpha_list[d]) for d in range(max_distance)]
    elif position == "n":
        offsets = [((0, d + 1), alpha_list[d]) for d in range(max_distance)]
    elif position == "s":
        offsets = [((0, -(d + 1)), alpha_list[d]) for d in range(max_distance)]
    elif position == "nw":
        offsets = [
            ((dx, dy), alpha_list[max(dx - 1, dy - 1)])
            for dx in range(1, max_distance + 1)
            for dy in range(1, max_distance + 1)
            if max(dx, dy) <= max_distance
        ]
    elif position == "ne":
        offsets = [
            ((-dx, dy), alpha_list[max(dx - 1, dy - 1)])
            for dx in range(1, max_distance + 1)
            for dy in range(1, max_distance + 1)
            if max(dx, dy) <= max_distance
        ]
    elif position == "sw":
        offsets = [
            ((dx, -dy), alpha_list[max(dx - 1, dy - 1)])
            for dx in range(1, max_distance + 1)
            for dy in range(1, max_distance + 1)
            if max(dx, dy) <= max_distance
        ]
    elif position == "se":
        offsets = [
            ((-dx, -dy), alpha_list[max(dx - 1, dy - 1)])
            for dx in range(1, max_distance + 1)
            for dy in range(1, max_distance + 1)
            if max(dx, dy) <= max_distance
        ]
    else:
        raise ValueError(f"unsupported position: {position}")

    projected = torch.zeros_like(mask)
    for (dx, dy), alpha in offsets:
        if alpha <= 0.0:
            continue
        shifted = _shift_mask(source, dx=dx, dy=dy)
        projected = torch.maximum(projected, shifted * alpha)
    return projected * mask


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

    # Normalize topology_mask here so callers can pass raw ComfyUI [B,H,W] masks
    if topology_mask is not None:
        topology_mask = normalize_mask(topology_mask, mask.shape[-2:]).to(device=mask.device, dtype=mask.dtype)
        if topology_mask.shape[0] not in (1, mask.shape[0]):
            raise ValueError("topology_mask batch must be 1 or match mask batch size")

    per_sample = []
    maps = []
    bbox_list = []
    present_union: set[Position] = set()
    global_denoise = float(max(0.0, min(1.0, global_denoise)))
    band_denoise_min = float(max(0.0, min(1.0, band_denoise_min)))
    band_denoise_max = float(max(0.0, min(1.0, band_denoise_max)))

    for idx in range(mask.shape[0]):
        sample_mask = mask[idx : idx + 1]
        try:
            bbox = mask_bbox(sample_mask)
        except RuntimeError:
            raise ValueError(
                f"mask sample {idx} is empty (all zeros) — "
                "at least one pixel must be non-zero to define a seam region"
            )
        bbox_list.append(bbox)
        x0, y0, x1, y1 = bbox
        bbox_w = max(x1 - x0, 1)
        bbox_h = max(y1 - y0, 1)

        topo_sample = None
        if topology_mask is not None:
            topo_sample = topology_mask[idx : idx + 1] if topology_mask.shape[0] > 1 else topology_mask

        present_positions = parse_present_positions(
            topo_sample,
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

        effective_positions = set(present_positions)
        if not effective_positions:
            if topology_mask is None:
                effective_positions = set(ALL_POSITIONS)
            else:
                target_map = base_map.clamp(0.0, 1.0)
                maps.append(target_map)
                per_sample.append(
                    {
                        "bbox": bbox,
                        "present_positions": tuple(sorted(present_positions)),
                        "uniform_fallback": False,
                    }
                )
                continue

        # When band covers the entire mask there is no room for a gradient;
        # treat the whole mask as "center" and apply band_denoise_max uniformly.
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
            if topology_mask is None:
                # Fast vectorized path: no topology, source = full binary zone
                zone = position_zone_mask(
                    sample_mask.shape[-2:], bbox, position,
                    device=sample_mask.device, dtype=sample_mask.dtype,
                )
                if float(zone.max().item()) <= 0.0:
                    continue
                weight = _zone_weight_from_bbox(
                    sample_mask, position, bbox,
                    band_width=int(band_width),
                    band_gradient_px=int(band_gradient_px),
                ).clamp(0.0, 1.0)
            else:
                # Projection path: topology provides non-uniform source weights
                zone = position_zone_mask(
                    sample_mask.shape[-2:], bbox, position,
                    device=sample_mask.device, dtype=sample_mask.dtype,
                )
                source = topo_sample * zone
                if float(source.max().item()) <= 0.0:
                    continue
                weight = _zone_weight(
                    sample_mask, source, position,
                    band_width=int(band_width),
                    band_gradient_px=int(band_gradient_px),
                ).clamp(0.0, 1.0)

            if float(weight.max().item()) <= 0.0:
                continue
            value = band_denoise_max - (band_denoise_max - band_denoise_min) * weight
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


def build_continuous_schedule(
    full_schedule: list[float],
    denoise: float,
    *,
    schedule_builder=None,
) -> list[float]:
    if not full_schedule:
        return []
    denoise = float(max(0.0, min(1.0, denoise)))
    steps = max(len(full_schedule) - 1, 0)
    if denoise >= 0.9999 or steps == 0:
        return [float(x) for x in full_schedule]
    if denoise <= 0.0:
        return [0.0]

    total_steps_float = float(steps) / denoise
    total_steps_int = int(math.ceil(total_steps_float))
    if schedule_builder is None:
        raise ValueError("schedule_builder is required when denoise < 1")
    dense_schedule = [float(x) for x in schedule_builder(total_steps_int)]

    start_offset = max(total_steps_float - float(steps), 0.0)
    positions = torch.linspace(start_offset, total_steps_float, steps + 1, dtype=torch.float64)
    out: list[float] = []
    for pos in positions.tolist():
        if pos >= total_steps_int:
            out.append(float(dense_schedule[-1]))
            continue
        low_idx = int(math.floor(pos))
        high_idx = min(low_idx + 1, total_steps_int)
        frac = float(pos - low_idx)
        low = float(dense_schedule[low_idx])
        high = float(dense_schedule[high_idx])
        out.append((1.0 - frac) * low + frac * high)
    out[-1] = float(dense_schedule[-1])
    return out


def apply_spatial_denoise_preservation(
    x: torch.Tensor,
    latent: torch.Tensor,
    noise: torch.Tensor,
    target_denoise_map: torch.Tensor,
    t_next: float,
) -> torch.Tensor:
    t_next = float(t_next)
    target = target_denoise_map.to(device=x.device, dtype=x.dtype).clamp(0.0, 1.0)
    if t_next <= 1e-6:
        return torch.where(target > 1e-6, x, latent.to(device=x.device, dtype=x.dtype))
    allowed = torch.minimum(target, torch.full_like(target, t_next))
    freedom = (allowed / t_next).clamp(0.0, 1.0)
    ref_x = (1.0 - allowed) * latent + allowed * noise
    return x * freedom + ref_x * (1.0 - freedom)

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt

from ..strip_ops import mask_bbox
from .merge_bands import build_seam_local_weight_map


Side = str
Position = str

SIDE_TOPOLOGY: dict[Side, Position] = {
    "left": "w",
    "right": "e",
    "top": "n",
    "bottom": "s",
}

TOPOLOGY_TO_SIDE: dict[Position, Side] = {v: k for k, v in SIDE_TOPOLOGY.items()}

CORNER_ADJACENT_SIDES: dict[Position, tuple[Position, Position]] = {
    "nw": ("n", "w"),
    "ne": ("n", "e"),
    "sw": ("s", "w"),
    "se": ("s", "e"),
}

SIDE_REPLACEMENT_CORNERS: dict[Position, tuple[Position, Position]] = {
    "n": ("nw", "ne"),
    "s": ("sw", "se"),
    "w": ("nw", "sw"),
    "e": ("ne", "se"),
}

CORNER_POSITIONS: tuple[Position, ...] = ("nw", "ne", "sw", "se")


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


def _normalize_mask_like(mask: torch.Tensor, reference_shape: tuple[int, int]) -> torch.Tensor:
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(1)
    elif mask.ndim != 4:
        raise ValueError("expected mask with shape [H,W], [B,H,W], or [B,1,H,W]")
    mask = mask.float()
    if mask.shape[-2:] != reference_shape:
        mask = F.interpolate(mask, size=reference_shape, mode="nearest")
    return (mask > 0.5).float()


def _build_sampling_generation_weight(
    mask: torch.Tensor,
    *,
    safety_ring_px: int,
) -> torch.Tensor:
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError("expected mask with shape [B,1,H,W]")
    ring = max(int(safety_ring_px), 0)
    if ring <= 0:
        return mask.clone()

    batch_weights = []
    for batch_index in range(mask.shape[0]):
        mask_np = mask[batch_index, 0].detach().cpu().numpy().astype(np.float32)
        outside = mask_np <= 0.5
        dist_to_inside = distance_transform_edt(outside).astype(np.float32)
        outside_weight = np.clip(((ring + 1.0) - dist_to_inside) / (ring + 1.0), 0.0, 1.0)
        generation_weight = np.where(mask_np > 0.5, 1.0, outside_weight).astype(np.float32)
        batch_weights.append(torch.from_numpy(generation_weight))

    stacked = torch.stack(batch_weights, dim=0).unsqueeze(1)
    return stacked.to(device=mask.device, dtype=mask.dtype)


def _build_low_freq_anchor_weight(
    mask: torch.Tensor,
    *,
    decay_px: int,
) -> torch.Tensor:
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError("expected mask with shape [B,1,H,W]")
    decay = max(int(decay_px), 1)
    batch_weights = []
    for batch_index in range(mask.shape[0]):
        mask_np = mask[batch_index, 0].detach().cpu().numpy().astype(np.float32)
        inside = mask_np > 0.5
        dist_in = distance_transform_edt(inside).astype(np.float32)
        weight = np.exp(-dist_in / float(decay)).astype(np.float32) * inside.astype(np.float32)
        batch_weights.append(torch.from_numpy(weight))
    stacked = torch.stack(batch_weights, dim=0).unsqueeze(1)
    return stacked.to(device=mask.device, dtype=mask.dtype)


def _position_zone_mask(
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


def _parse_present_positions(
    topology_mask: torch.Tensor | None,
    bbox: tuple[int, int, int, int],
    shape: tuple[int, int],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> set[Position]:
    if topology_mask is None:
        return set()
    norm = _normalize_mask_like(topology_mask, shape).to(device=device, dtype=dtype)
    present: set[Position] = set()
    for position in ("nw", "n", "ne", "w", "e", "sw", "s", "se"):
        zone = _position_zone_mask(shape, bbox, position, device=device, dtype=dtype)
        denom = float(zone.sum().item())
        if denom <= 0.0:
            continue
        coverage = float((norm * zone).sum().item() / denom)
        if coverage > 0.01:
            present.add(position)
    return present


def _local_corner_coords(
    bbox: tuple[int, int, int, int],
    corner: Position,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    x0, y0, x1, y1 = bbox
    bw = max(x1 - x0, 1)
    bh = max(y1 - y0, 1)
    xs = torch.arange(bw, device=device, dtype=dtype).view(1, 1, 1, bw)
    ys = torch.arange(bh, device=device, dtype=dtype).view(1, 1, bh, 1)
    if corner in {"nw", "sw"}:
        u = xs / max(bw - 1, 1)
    else:
        u = (float(bw - 1) - xs) / max(bw - 1, 1)
    if corner in {"nw", "ne"}:
        v = ys / max(bh - 1, 1)
    else:
        v = (float(bh - 1) - ys) / max(bh - 1, 1)
    return u.clamp(0.0, 1.0), v.clamp(0.0, 1.0)


def _make_bbox_map(
    bbox: tuple[int, int, int, int],
    inner_map: torch.Tensor,
    full_shape: tuple[int, int],
) -> torch.Tensor:
    height, width = full_shape
    out = torch.zeros(1, 1, height, width, device=inner_map.device, dtype=inner_map.dtype)
    x0, y0, x1, y1 = bbox
    out[:, :, y0:y1, x0:x1] = inner_map
    return out


def _extract_corner_patch(
    anchor_latent: torch.Tensor,
    bbox: tuple[int, int, int, int],
    corner: Position,
    width_px: int,
) -> torch.Tensor | None:
    _, _, height, width = anchor_latent.shape
    x0, y0, x1, y1 = bbox
    if corner == "nw":
        w = min(int(width_px), x0, y0)
        if w <= 0:
            return None
        return anchor_latent[:, :, y0 - w : y0, x0 - w : x0]
    if corner == "ne":
        w = min(int(width_px), width - x1, y0)
        if w <= 0:
            return None
        return anchor_latent[:, :, y0 - w : y0, x1 : x1 + w]
    if corner == "sw":
        w = min(int(width_px), x0, height - y1)
        if w <= 0:
            return None
        return anchor_latent[:, :, y1 : y1 + w, x0 - w : x0]
    if corner == "se":
        w = min(int(width_px), width - x1, height - y1)
        if w <= 0:
            return None
        return anchor_latent[:, :, y1 : y1 + w, x1 : x1 + w]
    raise ValueError(f"unsupported corner: {corner}")


def _extract_corner_stats(
    anchor_latent: torch.Tensor,
    bbox: tuple[int, int, int, int],
    corner: Position,
    width_px: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    patch = _extract_corner_patch(anchor_latent, bbox, corner, width_px)
    if patch is None or patch.numel() == 0:
        return None
    mean = patch.mean(dim=(-2, -1), keepdim=True)
    std = patch.std(dim=(-2, -1), keepdim=True, unbiased=False)
    return mean, std


def _build_side_share_map(
    bbox: tuple[int, int, int, int],
    side_pos: Position,
    corner: Position,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    x0, y0, x1, y1 = bbox
    bw = max(x1 - x0, 1)
    bh = max(y1 - y0, 1)
    if side_pos in {"n", "s"}:
        t = torch.linspace(0.0, 1.0, bw, device=device, dtype=dtype).view(1, 1, 1, bw)
        if corner in {"nw", "sw"}:
            return 1.0 - t
        return t
    t = torch.linspace(0.0, 1.0, bh, device=device, dtype=dtype).view(1, 1, bh, 1)
    if corner in {"nw", "ne"}:
        return 1.0 - t
    return t


def _build_low_freq_target_map(
    bbox: tuple[int, int, int, int],
    full_shape: tuple[int, int],
    side_profiles: dict[Side, torch.Tensor],
    corner_stats: dict[Position, tuple[torch.Tensor, torch.Tensor] | None],
    present_positions: set[Position],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    x0, y0, x1, y1 = bbox
    bw = max(x1 - x0, 1)
    bh = max(y1 - y0, 1)
    u = torch.linspace(0.0, 1.0, bw, device=device, dtype=dtype).view(1, 1, 1, bw)
    v = torch.linspace(0.0, 1.0, bh, device=device, dtype=dtype).view(1, 1, bh, 1)

    field_acc = None
    weight_acc = None

    if "left" in side_profiles and "right" in side_profiles:
        left = side_profiles["left"].to(device=device, dtype=dtype).expand(-1, -1, -1, bw)
        right = side_profiles["right"].to(device=device, dtype=dtype).expand(-1, -1, -1, bw)
        field = left * (1.0 - u) + right * u
        weight = torch.ones(1, 1, bh, bw, device=device, dtype=dtype)
        field_acc = field if field_acc is None else field_acc + field
        weight_acc = weight if weight_acc is None else weight_acc + weight
    elif "left" in side_profiles:
        field = side_profiles["left"].to(device=device, dtype=dtype).expand(-1, -1, -1, bw)
        weight = torch.ones(1, 1, bh, bw, device=device, dtype=dtype)
        field_acc = field if field_acc is None else field_acc + field
        weight_acc = weight if weight_acc is None else weight_acc + weight
    elif "right" in side_profiles:
        field = side_profiles["right"].to(device=device, dtype=dtype).expand(-1, -1, -1, bw)
        weight = torch.ones(1, 1, bh, bw, device=device, dtype=dtype)
        field_acc = field if field_acc is None else field_acc + field
        weight_acc = weight if weight_acc is None else weight_acc + weight

    if "top" in side_profiles and "bottom" in side_profiles:
        top = side_profiles["top"].to(device=device, dtype=dtype).expand(-1, -1, bh, -1)
        bottom = side_profiles["bottom"].to(device=device, dtype=dtype).expand(-1, -1, bh, -1)
        field = top * (1.0 - v) + bottom * v
        weight = torch.ones(1, 1, bh, bw, device=device, dtype=dtype)
        field_acc = field if field_acc is None else field_acc + field
        weight_acc = weight if weight_acc is None else weight_acc + weight
    elif "top" in side_profiles:
        field = side_profiles["top"].to(device=device, dtype=dtype).expand(-1, -1, bh, -1)
        weight = torch.ones(1, 1, bh, bw, device=device, dtype=dtype)
        field_acc = field if field_acc is None else field_acc + field
        weight_acc = weight if weight_acc is None else weight_acc + weight
    elif "bottom" in side_profiles:
        field = side_profiles["bottom"].to(device=device, dtype=dtype).expand(-1, -1, bh, -1)
        weight = torch.ones(1, 1, bh, bw, device=device, dtype=dtype)
        field_acc = field if field_acc is None else field_acc + field
        weight_acc = weight if weight_acc is None else weight_acc + weight

    corner_basis = {
        "nw": (1.0 - u) * (1.0 - v),
        "ne": u * (1.0 - v),
        "sw": (1.0 - u) * v,
        "se": u * v,
    }
    corner_field = None
    corner_weight = None
    for corner in CORNER_POSITIONS:
        stats = corner_stats.get(corner)
        if corner not in present_positions or stats is None:
            continue
        mean, _std = stats
        basis = corner_basis[corner]
        contrib = mean.to(device=device, dtype=dtype).expand(-1, -1, bh, bw) * basis
        corner_field = contrib if corner_field is None else corner_field + contrib
        corner_weight = basis if corner_weight is None else corner_weight + basis
    if corner_field is not None and corner_weight is not None:
        field = corner_field / corner_weight.clamp_min(1e-6)
        weight = torch.ones(1, 1, bh, bw, device=device, dtype=dtype)
        field_acc = field if field_acc is None else field_acc + field
        weight_acc = weight if weight_acc is None else weight_acc + weight

    if field_acc is None or weight_acc is None:
        return None
    inner = field_acc / weight_acc.clamp_min(1e-6)
    return _make_bbox_map(bbox, inner, full_shape)


def _corner_release_map(
    bbox: tuple[int, int, int, int],
    corner: Position,
    *,
    band_size: int,
    release_sides: tuple[Position, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    u, v = _local_corner_coords(bbox, corner, device=device, dtype=dtype)
    x0, y0, x1, y1 = bbox
    bw = max(x1 - x0, 1)
    bh = max(y1 - y0, 1)
    release = torch.ones(1, 1, bh, bw, device=device, dtype=dtype)
    frac_x = min(max(float(band_size) / max(bw, 1), 0.0), 0.95)
    frac_y = min(max(float(band_size) / max(bh, 1), 0.0), 0.95)
    if "w" in release_sides:
        release = release * ((u - frac_x) / max(1.0 - frac_x, 1e-6)).clamp(0.0, 1.0)
    if "e" in release_sides:
        release = release * ((u - frac_x) / max(1.0 - frac_x, 1e-6)).clamp(0.0, 1.0)
    if "n" in release_sides:
        release = release * ((v - frac_y) / max(1.0 - frac_y, 1e-6)).clamp(0.0, 1.0)
    if "s" in release_sides:
        release = release * ((v - frac_y) / max(1.0 - frac_y, 1e-6)).clamp(0.0, 1.0)
    return release


def _build_corner_wedge_weight(
    mask: torch.Tensor,
    bbox: tuple[int, int, int, int],
    corner: Position,
    *,
    band_size: int,
    adjacent_present: tuple[bool, bool],
) -> torch.Tensor:
    _, _, height, width = mask.shape
    device, dtype = mask.device, mask.dtype
    u, v = _local_corner_coords(bbox, corner, device=device, dtype=dtype)
    diagonal = (1.0 - (u - v).abs() / 0.35).clamp(0.0, 1.0).pow(2.0)
    radial = (1.0 - torch.maximum(u, v)).clamp(0.0, 1.0)
    inner = diagonal * radial
    side_a_present, side_b_present = adjacent_present
    adj_sides = CORNER_ADJACENT_SIDES[corner]
    if side_a_present and side_b_present:
        inner = inner * _corner_release_map(
            bbox,
            corner,
            band_size=band_size,
            release_sides=adj_sides,
            device=device,
            dtype=dtype,
        )
    elif side_a_present != side_b_present:
        present_side = adj_sides[0] if side_a_present else adj_sides[1]
        missing_side = adj_sides[1] if side_a_present else adj_sides[0]
        inner = inner * _corner_release_map(
            bbox,
            corner,
            band_size=band_size,
            release_sides=(present_side,),
            device=device,
            dtype=dtype,
        )
        if missing_side in {"n", "s"}:
            bias = (1.0 - v) if missing_side == "n" else (1.0 - v)
        else:
            bias = (1.0 - u) if missing_side == "w" else (1.0 - u)
        inner = inner * (0.25 + 0.75 * bias)
    return _make_bbox_map(bbox, inner, (height, width)) * mask


def _build_extra_contributions(
    anchor_latent: torch.Tensor,
    mask: torch.Tensor,
    bbox: tuple[int, int, int, int],
    present_positions: set[Position],
    *,
    anchor_width_px: int,
    anchor_falloff_px: int,
    process_left: bool,
    process_right: bool,
    process_top: bool,
    process_bottom: bool,
) -> list[dict]:
    contributions: list[dict] = []
    if not present_positions:
        return contributions
    process_enabled = {
        "w": bool(process_left),
        "e": bool(process_right),
        "n": bool(process_top),
        "s": bool(process_bottom),
    }
    corner_stats = {
        corner: _extract_corner_stats(anchor_latent, bbox, corner, anchor_width_px)
        for corner in CORNER_POSITIONS
        if corner in present_positions
    }
    total_present = len(present_positions)
    for side_pos, corners in SIDE_REPLACEMENT_CORNERS.items():
        if side_pos in present_positions or not process_enabled[side_pos]:
            continue
        available = [corner for corner in corners if corner in present_positions and corner_stats.get(corner) is not None]
        if not available:
            continue
        side_name = TOPOLOGY_TO_SIDE[side_pos]
        band_size = min(
            int(anchor_falloff_px),
            max((bbox[2] - bbox[0]) if side_name in {"left", "right"} else (bbox[3] - bbox[1]), 0),
        )
        if band_size <= 0:
            continue
        base_weight = torch.zeros_like(mask)
        if side_name == "top":
            base_weight[:, :, bbox[1] : bbox[1] + band_size, bbox[0] : bbox[2]] = 1.0
        elif side_name == "bottom":
            base_weight[:, :, bbox[3] - band_size : bbox[3], bbox[0] : bbox[2]] = 1.0
        elif side_name == "left":
            base_weight[:, :, bbox[1] : bbox[3], bbox[0] : bbox[0] + band_size] = 1.0
        else:
            base_weight[:, :, bbox[1] : bbox[3], bbox[2] - band_size : bbox[2]] = 1.0
        base_weight = base_weight * mask
        for corner in available:
            share = _build_side_share_map(bbox, side_pos, corner, device=mask.device, dtype=mask.dtype)
            share_map = torch.zeros_like(base_weight)
            if side_pos in {"n", "s"}:
                share_band = share.expand(-1, -1, band_size, -1)
                if side_name == "top":
                    share_map[:, :, bbox[1] : bbox[1] + band_size, bbox[0] : bbox[2]] = share_band
                else:
                    share_map[:, :, bbox[3] - band_size : bbox[3], bbox[0] : bbox[2]] = share_band
            else:
                share_band = share.expand(-1, -1, -1, band_size)
                if side_name == "left":
                    share_map[:, :, bbox[1] : bbox[3], bbox[0] : bbox[0] + band_size] = share_band
                else:
                    share_map[:, :, bbox[1] : bbox[3], bbox[2] - band_size : bbox[2]] = share_band
            mean, std = corner_stats[corner]
            contributions.append(
                {
                    "name": f"replace_{side_pos}_{corner}",
                    "weight": base_weight * share_map,
                    "mean": mean,
                    "std": std,
                }
            )

    for corner in CORNER_POSITIONS:
        stats = corner_stats.get(corner)
        if corner not in present_positions or stats is None:
            continue
        adj_a, adj_b = CORNER_ADJACENT_SIDES[corner]
        adj_present = (adj_a in present_positions, adj_b in present_positions)
        if total_present == 1 and not any(side in present_positions for side in ("n", "s", "w", "e")):
            weight = mask.clone()
        else:
            weight = _build_corner_wedge_weight(
                mask,
                bbox,
                corner,
                band_size=max(int(anchor_falloff_px), 1),
                adjacent_present=adj_present,
            )
        if float(weight.max().item()) <= 0.0:
            continue
        mean, std = stats
        contributions.append(
            {
                "name": f"corner_{corner}",
                "weight": weight,
                "mean": mean,
                "std": std,
            }
        )
    return contributions


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
    topology_mask: torch.Tensor | None = None,
    anchor_width_px: int,
    anchor_falloff_px: int,
    process_left: bool = True,
    process_right: bool = True,
    process_top: bool = True,
    process_bottom: bool = True,
    reduce: str = "mean",
    profile_smooth_kernel: int = 5,
    safety_ring_px: int = 0,
    low_freq_anchor_decay_px: int = 64,
) -> dict:
    if anchor_latent.ndim != 4:
        raise ValueError("expected anchor_latent with shape [B,C,H,W]")
    mask = _normalize_mask_like(mask, anchor_latent.shape[-2:])
    bbox = mask_bbox(mask)
    x0, y0, x1, y1 = bbox
    present_positions = _parse_present_positions(
        topology_mask,
        bbox,
        anchor_latent.shape[-2:],
        device=anchor_latent.device,
        dtype=anchor_latent.dtype,
    )

    sides: list[Side] = []
    if process_left and x0 > 0 and (not present_positions or "w" in present_positions):
        sides.append("left")
    if process_right and x1 < anchor_latent.shape[-1] and (not present_positions or "e" in present_positions):
        sides.append("right")
    if process_top and y0 > 0 and (not present_positions or "n" in present_positions):
        sides.append("top")
    if process_bottom and y1 < anchor_latent.shape[-2] and (not present_positions or "s" in present_positions):
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

    corner_stats = {
        corner: _extract_corner_stats(anchor_latent, bbox, corner, anchor_width_px)
        for corner in CORNER_POSITIONS
        if corner in present_positions
    }
    extra_contributions = _build_extra_contributions(
        anchor_latent,
        mask,
        bbox,
        present_positions,
        anchor_width_px=anchor_width_px,
        anchor_falloff_px=anchor_falloff_px,
        process_left=process_left,
        process_right=process_right,
        process_top=process_top,
        process_bottom=process_bottom,
    )
    generation_weight = _build_sampling_generation_weight(
        mask,
        safety_ring_px=safety_ring_px,
    )
    low_freq_target = _build_low_freq_target_map(
        bbox,
        anchor_latent.shape[-2:],
        per_side_profiles,
        corner_stats,
        present_positions,
        device=anchor_latent.device,
        dtype=anchor_latent.dtype,
    )
    low_freq_weight = _build_low_freq_anchor_weight(
        mask,
        decay_px=low_freq_anchor_decay_px,
    ) if low_freq_target is not None else None

    return {
        "bbox": bbox,
        "mask": mask,
        "generation_weight": generation_weight,
        "low_freq_target": low_freq_target,
        "low_freq_weight": low_freq_weight,
        "sides": tuple(per_side_profiles.keys()),
        "present_positions": tuple(sorted(present_positions)),
        "profiles": per_side_profiles,
        "std_profiles": per_side_std_profiles,
        "weights": per_side_weights,
        "band_sizes": per_side_band_sizes,
        "sample_widths": per_side_sample_widths,
        "extra_contributions": extra_contributions,
        "reduce": reduce,
    }


def apply_seam_anchor_correction(
    denoised: torch.Tensor,
    anchor_state: dict,
    effective_strength: float,
    *,
    low_freq_strength: float = 0.0,
) -> torch.Tensor:
    if (effective_strength <= 0.0 and low_freq_strength <= 0.0) or (
        not anchor_state.get("sides")
        and not anchor_state.get("extra_contributions")
        and anchor_state.get("low_freq_target") is None
    ):
        return denoised

    bbox = anchor_state["bbox"]
    reduce = anchor_state["reduce"]
    profiles = anchor_state["profiles"]
    weights = anchor_state["weights"]
    band_sizes = anchor_state["band_sizes"]
    sample_widths = anchor_state["sample_widths"]
    extra_contributions = anchor_state.get("extra_contributions", [])

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
        corrected = denoised
    else:
        coverage = total_weight.clamp(0.0, 1.0)
        merged = torch.where(total_weight > 1.0, weighted_sum / total_weight.clamp_min(1e-8), weighted_sum)
        merged = torch.where(total_weight > 1.0, merged * coverage, merged)
        corrected = denoised + merged * float(effective_strength)
    if not extra_contributions:
        guided = corrected
    else:
        weighted_sum = torch.zeros_like(corrected)
        total_weight = torch.zeros(corrected.shape[0], 1, corrected.shape[-2], corrected.shape[-1], device=corrected.device, dtype=corrected.dtype)
        for contribution in extra_contributions:
            weight = _match_batch(contribution["weight"], corrected.shape[0]).to(device=corrected.device, dtype=corrected.dtype)
            target_mean = _match_batch(contribution["mean"], corrected.shape[0]).to(device=corrected.device, dtype=corrected.dtype)
            target_mean = _match_channels(target_mean, corrected.shape[1]).expand_as(corrected)
            weighted_sum = weighted_sum + (target_mean - corrected) * weight
            total_weight = total_weight + weight
        if float(total_weight.max().item()) <= 0.0:
            guided = corrected
        else:
            coverage = total_weight.clamp(0.0, 1.0)
            merged = torch.where(total_weight > 1.0, weighted_sum / total_weight.clamp_min(1e-8), weighted_sum)
            merged = torch.where(total_weight > 1.0, merged * coverage, merged)
            guided = corrected + merged * float(effective_strength)

    if low_freq_strength <= 0.0:
        return guided
    low_freq_target = anchor_state.get("low_freq_target")
    low_freq_weight = anchor_state.get("low_freq_weight")
    if low_freq_target is None or low_freq_weight is None:
        return guided
    target = _match_batch(low_freq_target, guided.shape[0]).to(device=guided.device, dtype=guided.dtype)
    target = _match_channels(target, guided.shape[1])
    weight = _match_batch(low_freq_weight, guided.shape[0]).to(device=guided.device, dtype=guided.dtype)
    if target.shape[-2:] != guided.shape[-2:]:
        target = F.interpolate(target, size=guided.shape[-2:], mode="bilinear", align_corners=False)
    if weight.shape[-2:] != guided.shape[-2:]:
        weight = F.interpolate(weight, size=guided.shape[-2:], mode="bilinear", align_corners=False)
    return guided + (target - guided) * weight * float(low_freq_strength)


def apply_seam_latent_guidance(
    x: torch.Tensor,
    anchor_state: dict,
    effective_strength: float,
    *,
    mode: str = "mean_shift",
    boundary_only: bool = False,
    variance_limit: float = 2.0,
    low_freq_strength: float = 0.0,
) -> torch.Tensor:
    if (effective_strength <= 0.0 and low_freq_strength <= 0.0) or (
        not anchor_state.get("sides")
        and not anchor_state.get("extra_contributions")
        and anchor_state.get("low_freq_target") is None
    ):
        return x

    bbox = anchor_state["bbox"]
    reduce = anchor_state["reduce"]
    profiles = anchor_state["profiles"]
    std_profiles = anchor_state["std_profiles"]
    weights = anchor_state["weights"]
    band_sizes = anchor_state["band_sizes"]
    sample_widths = anchor_state["sample_widths"]
    extra_contributions = anchor_state.get("extra_contributions", [])

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
        guided = x
    else:
        coverage = total_weight.clamp(0.0, 1.0)
        merged = torch.where(total_weight > 1.0, weighted_sum / total_weight.clamp_min(1e-8), weighted_sum)
        merged = torch.where(total_weight > 1.0, merged * coverage, merged)
        guided = x + merged * float(effective_strength)
    if extra_contributions:
        weighted_sum = torch.zeros_like(guided)
        total_weight = torch.zeros(guided.shape[0], 1, guided.shape[-2], guided.shape[-1], device=guided.device, dtype=guided.dtype)
        for contribution in extra_contributions:
            weight = _match_batch(contribution["weight"], guided.shape[0]).to(device=guided.device, dtype=guided.dtype)
            if boundary_only:
                weight = weight.clamp(0.0, 1.0).pow(2.0)
            target_mean = _match_batch(contribution["mean"], guided.shape[0]).to(device=guided.device, dtype=guided.dtype)
            target_mean = _match_channels(target_mean, guided.shape[1]).expand_as(guided)
            weighted_sum = weighted_sum + (target_mean - guided) * weight
            total_weight = total_weight + weight
        if float(total_weight.max().item()) > 0.0:
            coverage = total_weight.clamp(0.0, 1.0)
            merged = torch.where(total_weight > 1.0, weighted_sum / total_weight.clamp_min(1e-8), weighted_sum)
            merged = torch.where(total_weight > 1.0, merged * coverage, merged)
            guided = guided + merged * float(effective_strength)
    if low_freq_strength <= 0.0:
        return guided
    low_freq_target = anchor_state.get("low_freq_target")
    low_freq_weight = anchor_state.get("low_freq_weight")
    if low_freq_target is None or low_freq_weight is None:
        return guided
    target = _match_batch(low_freq_target, guided.shape[0]).to(device=guided.device, dtype=guided.dtype)
    target = _match_channels(target, guided.shape[1])
    weight = _match_batch(low_freq_weight, guided.shape[0]).to(device=guided.device, dtype=guided.dtype)
    if target.shape[-2:] != guided.shape[-2:]:
        target = F.interpolate(target, size=guided.shape[-2:], mode="bilinear", align_corners=False)
    if weight.shape[-2:] != guided.shape[-2:]:
        weight = F.interpolate(weight, size=guided.shape[-2:], mode="bilinear", align_corners=False)
    return guided + (target - guided) * weight * float(low_freq_strength)

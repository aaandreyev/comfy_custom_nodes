"""Seam-profile tone correction: match the inside of a stitch boundary to the outside.

Unlike the neighbor tone match LUT — which learns a colour->correction mapping from
a donor ring outside the mask and therefore can only ever remove drift that is
present in that ring — this correction measures the tone step across the stitch
seam itself and cancels it. Per seam location, both sides' low-passed tone is
linearly extrapolated to the seam line (two sampling bands per side, so natural
cross-seam gradients cancel), their difference is smoothed along the contour,
then applied inside the mask with weight 1.0 at the seam fading to 0 over
``inner_width``.

The seam is parameterised by nearest-contour-pixel assignment (no contour
ordering), so arbitrary mask topology works: multiple blobs, holes, masks
touching the frame edge. Masks thinner than ``2 * arc_smooth_px`` will mix the
profiles of their opposite sides — intended for blob-like inpaint regions.

All sampling runs on CPU float32 (scipy); inputs may live on any device/dtype.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt, grey_erosion

from .fast_filters import gaussian, gaussian_stack

from .neighbor_tone_match import (
    _compress_to_unit_gamut,
    _freeform_inner_weight,
    _linear_to_srgb,
    _rgb_to_yuv,
    _srgb_to_linear,
    _yuv_to_rgb,
)
from .seam_latent_anchor import SIDE_TOPOLOGY, _parse_present_positions
from ..strip_ops import mask_bbox

_EPS = 1e-4


def _normalized_lowpass(values: np.ndarray, support: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian average of ``values`` restricted to ``support``, extended smoothly outside it."""
    den = gaussian(support, sigma)
    num = gaussian_stack(values, support, sigma)
    return num / np.maximum(den, _EPS)[None], den


def _extrapolate_to_boundary(
    values: np.ndarray,
    side: np.ndarray,
    dist: np.ndarray,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Linear extrapolation of one side's tone to the seam line itself.

    Sampling a single band on each side would report the natural cross-seam
    gradient as a fake step (band centroids sit a few px apart). Two bands per
    side plus first-order extrapolation to distance 0 cancel that bias exactly
    for locally linear content.
    """
    near_px = max(2.0, round(2.0 * sigma))
    far_px = max(near_px + 2.0, round(5.0 * sigma))
    near = (side & (dist > 0) & (dist <= near_px)).astype(np.float32)
    far = (side & (dist > near_px) & (dist <= far_px)).astype(np.float32)
    v_near, den_near = _normalized_lowpass(values, near, sigma)
    v_far, den_far = _normalized_lowpass(values, far, sigma)
    d_near = gaussian(dist * near, sigma) / np.maximum(den_near, _EPS)
    d_far = gaussian(dist * far, sigma) / np.maximum(den_far, _EPS)
    ok = (den_near > _EPS) & (den_far > _EPS) & ((d_far - d_near) > 0.5)
    slope = (v_near - v_far) / np.maximum(d_far - d_near, 0.5)[None]
    v0 = np.where(ok[None], v_near + slope * d_near[None], v_near)
    return v0, den_near > _EPS


def _seam_delta_field(
    ref_yuv: np.ndarray,
    img_yuv: np.ndarray,
    seam_mask: np.ndarray,
    *,
    lowpass_sigma: float,
    arc_smooth_px: float,
    outside_support: np.ndarray | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None, dict]:
    """Per-pixel additive YUV delta that cancels the cross-seam tone step.

    Returns the delta sampled at the nearest valid contour pixel for every pixel
    of the crop (multiply by an inward falloff before use) plus the valid-contour
    map, or (None, None, meta) when the seam has no usable contour.
    ``outside_support`` optionally restricts where reference tone may be sampled
    (e.g. only sides that have a real neighbor).
    """
    dist_in = distance_transform_edt(seam_mask)
    dist_out = distance_transform_edt(~seam_mask)
    contour = seam_mask & (dist_in <= 1.5)
    if not contour.any():
        return None, None, {"reason": "no_contour"}

    outside = ~seam_mask if outside_support is None else ((~seam_mask) & outside_support)
    ref_at_seam, ok_out = _extrapolate_to_boundary(ref_yuv, outside, dist_out, lowpass_sigma)
    img_at_seam, ok_in = _extrapolate_to_boundary(img_yuv, seam_mask, dist_in, lowpass_sigma)
    valid = contour & ok_out & ok_in
    if not valid.any():
        return None, None, {"reason": "no_valid_contour_samples"}

    delta_raw = ref_at_seam - img_at_seam
    support = valid.astype(np.float32)
    den_arc = gaussian(support, arc_smooth_px)
    num_arc = gaussian_stack(delta_raw[:3], support, arc_smooth_px)
    delta_line = num_arc / np.maximum(den_arc, _EPS)[None]

    nearest = distance_transform_edt(~valid, return_indices=True, return_distances=False)
    delta = delta_line[:, nearest[0], nearest[1]]
    meta = {
        "contour_px": int(valid.sum()),
        "delta_line_mean_abs_y": float(np.abs(delta_line[0][valid]).mean()),
        "delta_line_max_abs_y": float(np.abs(delta_line[0][valid]).max()),
    }
    return delta, valid, meta


def apply_seam_profile_tone_match(
    reference_rgb: torch.Tensor,
    image_rgb: torch.Tensor,
    mask: torch.Tensor,
    *,
    inner_width: int,
    inner_flat_top_px: int = 0,
    seam_inset_px: int = 0,
    lowpass_sigma: float = 3.0,
    arc_smooth_px: float = 12.0,
    max_correction: float = 0.5,
    luma_strength: float = 1.0,
    chroma_strength: float = 1.0,
    u_strength: float | None = None,
    v_strength: float | None = None,
    color_space: str = "srgb",
    yuv_matrix: str = "bt709",
) -> tuple[torch.Tensor, dict]:
    if reference_rgb.ndim != 4 or image_rgb.ndim != 4:
        raise ValueError("reference_rgb and image_rgb must be BCHW")
    if reference_rgb.shape[1:] != image_rgb.shape[1:]:
        raise ValueError("reference_rgb and image_rgb must have the same C,H,W")
    if reference_rgb.shape[0] == 1 and image_rgb.shape[0] > 1:
        reference_rgb = reference_rgb.expand(image_rgb.shape[0], -1, -1, -1)
    elif reference_rgb.shape[0] != image_rgb.shape[0]:
        raise ValueError("reference batch must be 1 or match the image batch")

    image_dtype = image_rgb.dtype
    out_device = image_rgb.device
    u_strength = float(chroma_strength if u_strength is None else u_strength)
    v_strength = float(chroma_strength if v_strength is None else v_strength)

    mask = mask.to(dtype=torch.float32)
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    elif mask.ndim != 4:
        raise ValueError("mask must be [B,H,W] or [B,1,H,W]")
    if mask.shape[-2:] != image_rgb.shape[-2:]:
        mask = torch.nn.functional.interpolate(mask, size=image_rgb.shape[-2:], mode="bilinear", align_corners=False)
    soft_mask = mask.clamp(0.0, 1.0)
    if soft_mask.shape[0] == 1 and image_rgb.shape[0] > 1:
        soft_mask = soft_mask.expand(image_rgb.shape[0], -1, -1, -1)

    channel_scale = np.array([float(luma_strength), u_strength, v_strength], dtype=np.float32).reshape(3, 1, 1)
    corrected = image_rgb.clone()
    debug_items: list[dict] = []
    pad = int(math.ceil(3.0 * max(lowpass_sigma, arc_smooth_px))) + 2
    height, width = image_rgb.shape[-2:]

    for idx in range(image_rgb.shape[0]):
        mask_np = soft_mask[idx, 0].detach().float().cpu().numpy()
        mask_bool = mask_np > 0.5
        if seam_inset_px > 0:
            size = 2 * int(seam_inset_px) + 1
            mask_bool = grey_erosion(mask_bool.astype(np.float32), size=(size, size)) > 0.5
        if not mask_bool.any():
            debug_items.append({"reason": "empty_mask"})
            continue

        ys, xs = np.where(mask_bool)
        cy0 = max(int(ys.min()) - pad, 0)
        cy1 = min(int(ys.max()) + 1 + pad, height)
        cx0 = max(int(xs.min()) - pad, 0)
        cx1 = min(int(xs.max()) + 1 + pad, width)
        ref_crop = reference_rgb[idx : idx + 1, :, cy0:cy1, cx0:cx1].detach().float().cpu()
        img_crop = image_rgb[idx : idx + 1, :, cy0:cy1, cx0:cx1].detach().float().cpu()
        seam_crop = mask_bool[cy0:cy1, cx0:cx1]

        if color_space == "srgb":
            ref_yuv_t = _rgb_to_yuv(_srgb_to_linear(ref_crop), matrix=yuv_matrix)
            img_yuv_t = _rgb_to_yuv(_srgb_to_linear(img_crop), matrix=yuv_matrix)
        else:
            ref_yuv_t = _rgb_to_yuv(ref_crop, matrix=yuv_matrix)
            img_yuv_t = _rgb_to_yuv(img_crop, matrix=yuv_matrix)
        ref_yuv = ref_yuv_t[0].numpy()
        img_yuv = img_yuv_t[0].numpy()

        delta, _valid, meta = _seam_delta_field(
            ref_yuv,
            img_yuv,
            seam_crop,
            lowpass_sigma=float(lowpass_sigma),
            arc_smooth_px=float(arc_smooth_px),
        )
        if delta is None:
            meta["bbox"] = (cx0, cy0, cx1, cy1)
            debug_items.append(meta)
            continue

        limit = abs(float(max_correction))
        delta = np.clip(delta, -limit, limit) * channel_scale
        dist_in = distance_transform_edt(seam_crop)
        weight = _freeform_inner_weight(
            dist_in,
            inner_width=int(inner_width),
            flat_top_px=int(inner_flat_top_px),
        )
        weight = weight * (seam_crop & (dist_in <= float(max(int(inner_width), 1)))).astype(np.float32)

        corrected_yuv = img_yuv_t + torch.from_numpy(delta * weight[None]).float().unsqueeze(0)
        neutral = corrected_yuv[:, :1].clamp(0.0, 1.0).expand(-1, 3, -1, -1)
        corrected_lin = _compress_to_unit_gamut(_yuv_to_rgb(corrected_yuv, matrix=yuv_matrix), neutral)
        if color_space == "srgb":
            corrected_rgb = _linear_to_srgb(corrected_lin).clamp(0.0, 1.0)
        else:
            corrected_rgb = corrected_lin.clamp(0.0, 1.0)

        seam_t = torch.from_numpy(seam_crop.astype(np.float32)).view(1, 1, *seam_crop.shape)
        corrected_rgb = corrected_rgb * seam_t + img_crop * (1.0 - seam_t)
        corrected[idx : idx + 1, :, cy0:cy1, cx0:cx1] = corrected_rgb.to(dtype=image_dtype, device=out_device)
        meta.update({
            "reason": "applied",
            "bbox": (cx0, cy0, cx1, cy1),
            "seam_px": int(seam_crop.sum()),
        })
        debug_items.append(meta)

    applied = any(item.get("reason") == "applied" for item in debug_items)
    return corrected, {
        "reason": "applied" if applied else (debug_items[0].get("reason", "empty_batch") if debug_items else "empty_batch"),
        "per_sample": debug_items,
        "soft_mask": soft_mask,
    }


def apply_sided_seam_profile_tone_match(
    reference_rgb: torch.Tensor,
    image_rgb: torch.Tensor,
    mask: torch.Tensor,
    topology_mask: torch.Tensor | None = None,
    *,
    inner_width: int,
    inner_flat_top_px: int = 0,
    seam_inset_px: int = 0,
    process_left: bool = True,
    process_right: bool = True,
    process_top: bool = True,
    process_bottom: bool = True,
    lowpass_sigma: float = 3.0,
    arc_smooth_px: float = 12.0,
    max_correction: float = 0.5,
    luma_strength: float = 1.0,
    chroma_strength: float = 1.0,
    u_strength: float | None = None,
    v_strength: float | None = None,
    color_space: str = "srgb",
    yuv_matrix: str = "bt709",
) -> tuple[torch.Tensor, dict]:
    """Seam-profile correction for the outpaint/edit placement of NeighborToneMatch.

    Same contract as ``apply_neighbor_tone_match`` (side flags + topology mask
    decide which bbox sides carry a real neighbor), but the correction is the
    measured cross-seam tone step instead of a donor-band LUT: reference tone is
    sampled only outside the enabled sides, and the inward falloff is measured
    from the valid seam, so sides without a neighbor are never corrected against.
    """
    if reference_rgb.ndim != 4 or image_rgb.ndim != 4:
        raise ValueError("reference_rgb and image_rgb must be BCHW")
    if reference_rgb.shape[1:] != image_rgb.shape[1:]:
        raise ValueError("reference_rgb and image_rgb must have the same C,H,W")
    if reference_rgb.shape[0] == 1 and image_rgb.shape[0] > 1:
        reference_rgb = reference_rgb.expand(image_rgb.shape[0], -1, -1, -1)
    elif reference_rgb.shape[0] != image_rgb.shape[0]:
        raise ValueError("reference batch must be 1 or match the image batch")

    image_dtype = image_rgb.dtype
    out_device = image_rgb.device
    u_strength = float(chroma_strength if u_strength is None else u_strength)
    v_strength = float(chroma_strength if v_strength is None else v_strength)

    mask = mask.to(dtype=torch.float32)
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    elif mask.ndim != 4:
        raise ValueError("mask must be [B,H,W] or [B,1,H,W]")
    if mask.shape[-2:] != image_rgb.shape[-2:]:
        mask = torch.nn.functional.interpolate(mask, size=image_rgb.shape[-2:], mode="bilinear", align_corners=False)
    soft_mask = mask.clamp(0.0, 1.0)
    if soft_mask.shape[0] == 1 and image_rgb.shape[0] > 1:
        soft_mask = soft_mask.expand(image_rgb.shape[0], -1, -1, -1)

    height, width = image_rgb.shape[-2:]
    try:
        bbox = mask_bbox((soft_mask > 1e-3).to(dtype=torch.float32))
    except RuntimeError:
        return image_rgb, {"reason": "empty_mask", "per_sample": [], "soft_mask": soft_mask}
    x0, y0, x1, y1 = bbox
    present_positions = _parse_present_positions(
        topology_mask,
        bbox,
        (int(height), int(width)),
        device=soft_mask.device,
        dtype=torch.float32,
    )
    sides: list[str] = []
    if process_left and x0 > 0 and (not present_positions or SIDE_TOPOLOGY["left"] in present_positions):
        sides.append("left")
    if process_right and x1 < width and (not present_positions or SIDE_TOPOLOGY["right"] in present_positions):
        sides.append("right")
    if process_top and y0 > 0 and (not present_positions or SIDE_TOPOLOGY["top"] in present_positions):
        sides.append("top")
    if process_bottom and y1 < height and (not present_positions or SIDE_TOPOLOGY["bottom"] in present_positions):
        sides.append("bottom")
    if not sides:
        return image_rgb, {
            "reason": "no_processable_sides",
            "per_sample": [],
            "bbox": bbox,
            "present_positions": tuple(sorted(present_positions)),
            "soft_mask": soft_mask,
        }

    channel_scale = np.array([float(luma_strength), u_strength, v_strength], dtype=np.float32).reshape(3, 1, 1)
    corrected = image_rgb.clone()
    debug_items: list[dict] = []
    pad = int(math.ceil(3.0 * max(lowpass_sigma, arc_smooth_px))) + 2

    for idx in range(image_rgb.shape[0]):
        mask_np = soft_mask[idx, 0].detach().float().cpu().numpy()
        mask_bool = mask_np > 0.5
        if seam_inset_px > 0:
            size = 2 * int(seam_inset_px) + 1
            mask_bool = grey_erosion(mask_bool.astype(np.float32), size=(size, size)) > 0.5
        if not mask_bool.any():
            debug_items.append({"reason": "empty_mask"})
            continue

        ys, xs = np.where(mask_bool)
        cy0 = max(int(ys.min()) - pad, 0)
        cy1 = min(int(ys.max()) + 1 + pad, height)
        cx0 = max(int(xs.min()) - pad, 0)
        cx1 = min(int(xs.max()) + 1 + pad, width)
        ref_crop = reference_rgb[idx : idx + 1, :, cy0:cy1, cx0:cx1].detach().float().cpu()
        img_crop = image_rgb[idx : idx + 1, :, cy0:cy1, cx0:cx1].detach().float().cpu()
        seam_crop = mask_bool[cy0:cy1, cx0:cx1]

        bx0 = int(xs.min()) - cx0
        bx1 = int(xs.max()) + 1 - cx0
        by0 = int(ys.min()) - cy0
        by1 = int(ys.max()) + 1 - cy0
        ch, cw = seam_crop.shape
        yy, xx = np.mgrid[0:ch, 0:cw]
        support = np.zeros((ch, cw), dtype=bool)
        if "left" in sides:
            support |= xx < bx0
        if "right" in sides:
            support |= xx >= bx1
        if "top" in sides:
            support |= yy < by0
        if "bottom" in sides:
            support |= yy >= by1

        if color_space == "srgb":
            ref_yuv_t = _rgb_to_yuv(_srgb_to_linear(ref_crop), matrix=yuv_matrix)
            img_yuv_t = _rgb_to_yuv(_srgb_to_linear(img_crop), matrix=yuv_matrix)
        else:
            ref_yuv_t = _rgb_to_yuv(ref_crop, matrix=yuv_matrix)
            img_yuv_t = _rgb_to_yuv(img_crop, matrix=yuv_matrix)

        delta, valid, meta = _seam_delta_field(
            ref_yuv_t[0].numpy(),
            img_yuv_t[0].numpy(),
            seam_crop,
            lowpass_sigma=float(lowpass_sigma),
            arc_smooth_px=float(arc_smooth_px),
            outside_support=support,
        )
        if delta is None:
            meta["bbox"] = (cx0, cy0, cx1, cy1)
            debug_items.append(meta)
            continue

        limit = abs(float(max_correction))
        delta = np.clip(delta, -limit, limit) * channel_scale
        dist_valid = distance_transform_edt(~valid)
        weight = _freeform_inner_weight(
            dist_valid,
            inner_width=int(inner_width),
            flat_top_px=int(inner_flat_top_px),
        )
        weight = weight * (seam_crop & (dist_valid <= float(max(int(inner_width), 1)))).astype(np.float32)

        corrected_yuv = img_yuv_t + torch.from_numpy(delta * weight[None]).float().unsqueeze(0)
        neutral = corrected_yuv[:, :1].clamp(0.0, 1.0).expand(-1, 3, -1, -1)
        corrected_lin = _compress_to_unit_gamut(_yuv_to_rgb(corrected_yuv, matrix=yuv_matrix), neutral)
        if color_space == "srgb":
            corrected_rgb = _linear_to_srgb(corrected_lin).clamp(0.0, 1.0)
        else:
            corrected_rgb = corrected_lin.clamp(0.0, 1.0)

        seam_t = torch.from_numpy(seam_crop.astype(np.float32)).view(1, 1, ch, cw)
        corrected_rgb = corrected_rgb * seam_t + img_crop * (1.0 - seam_t)
        corrected[idx : idx + 1, :, cy0:cy1, cx0:cx1] = corrected_rgb.to(dtype=image_dtype, device=out_device)
        meta.update({
            "reason": "applied",
            "bbox": (cx0, cy0, cx1, cy1),
            "sides": list(sides),
            "seam_px": int(seam_crop.sum()),
        })
        debug_items.append(meta)

    applied = any(item.get("reason") == "applied" for item in debug_items)
    return corrected, {
        "reason": "applied" if applied else (debug_items[0].get("reason", "empty_batch") if debug_items else "empty_batch"),
        "per_sample": debug_items,
        "present_positions": tuple(sorted(present_positions)),
        "soft_mask": soft_mask,
    }

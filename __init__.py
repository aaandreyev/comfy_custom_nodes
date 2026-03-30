"""
Unified ComfyUI custom nodes:
- PoissonInpaintPrefill
- MaskHarmonize
"""

from collections import OrderedDict

import cv2
import numpy as np
import torch
from scipy.fft import dstn
from scipy.ndimage import distance_transform_edt, label


MAX_HARMONIZE_WORKDIM = 640
_FILL_GEOMETRY_CACHE: "OrderedDict[tuple, dict]" = OrderedDict()
_FILL_GEOMETRY_CACHE_MAX = 4


def _resize_array(src: np.ndarray, width: int, height: int, interpolation: int) -> np.ndarray:
    return cv2.resize(src, (width, height), interpolation=interpolation)


def _fast_distance_transform(mask_bin: np.ndarray) -> np.ndarray:
    return cv2.distanceTransform(mask_bin.astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)


def _distance_to_seed_map(seed: np.ndarray) -> np.ndarray:
    if not np.any(seed):
        return np.full(seed.shape, np.inf, dtype=np.float32)
    return cv2.distanceTransform((1 - seed.astype(np.uint8)), cv2.DIST_L2, 5).astype(np.float32)


def _to_lab(img_u8: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_u8.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB).astype(np.float32)


def _lab_to_rgb_u8(lab: np.ndarray) -> np.ndarray:
    clipped = np.clip(lab, [0, -128, -128], [100, 127, 127]).astype(np.float32)
    return np.clip(cv2.cvtColor(clipped, cv2.COLOR_LAB2RGB) * 255.0, 0, 255).astype(np.uint8)


def _gaussian_blur_aniso(src: np.ndarray, sigma_x: float, sigma_y: float) -> np.ndarray:
    sigma_x = max(float(sigma_x), 1e-3)
    sigma_y = max(float(sigma_y), 1e-3)
    return cv2.GaussianBlur(np.asarray(src, dtype=np.float32), (0, 0), sigmaX=sigma_x, sigmaY=sigma_y)


def _gaussian_distance_weight(dist: np.ndarray, sigma: float) -> np.ndarray:
    sigma = max(float(sigma), 1e-3)
    weight = np.exp(-0.5 * np.square(dist.astype(np.float32) / sigma))
    weight[~np.isfinite(dist)] = 0.0
    return weight.astype(np.float32)


def _compute_harmonize_delta(
    img_u8: np.ndarray,
    mask_np: np.ndarray,
    *,
    strip_width: int,
    blur_sigma: float,
    mask_threshold: float,
    alpha_np=None,
    protect_mask_np=None,
    corner_spread: int = 0,
):
    h, w = img_u8.shape[:2]
    lab = _to_lab(img_u8)

    mask_f = np.clip(mask_np.astype(np.float32), 0.0, 1.0)
    mask_bin = (mask_f >= mask_threshold).astype(np.uint8)
    if mask_bin.sum() == 0 or mask_bin.sum() == h * w:
        return None

    dist_in = _fast_distance_transform(mask_bin)
    dist_out = _fast_distance_transform(1 - mask_bin)
    inner_strip = (mask_bin == 1) & (dist_in > 0) & (dist_in <= strip_width)
    outer_strip = (mask_bin == 0) & (dist_out > 0) & (dist_out <= strip_width)
    if inner_strip.sum() == 0 or outer_strip.sum() == 0:
        return None

    effective_sigma = max(float(blur_sigma), 0.5)
    strip_cross_sigma = max(float(strip_width) * 0.35, 0.75)
    corner_mix_sigma = (
        float(corner_spread)
        if corner_spread > 0
        else max(float(strip_width) * 0.35, 2.0)
    )
    alpha_valid = (
        np.clip(alpha_np.astype(np.float32), 0.0, 1.0) if alpha_np is not None
        else np.ones((h, w), dtype=np.float32)
    )
    protect_valid = (
        1.0 - np.clip(protect_mask_np.astype(np.float32), 0.0, 1.0)
        if protect_mask_np is not None
        else np.ones((h, w), dtype=np.float32)
    )
    pixel_valid = np.clip(alpha_valid * protect_valid, 0.0, 1.0)

    eps = 1e-4
    inner_edge = inner_strip & (dist_in <= 1.5)
    outer_edge = outer_strip & (dist_out <= 1.5)

    outside_up = np.zeros((h, w), dtype=bool)
    outside_up[1:, :] = mask_bin[:-1, :] == 0
    outside_down = np.zeros((h, w), dtype=bool)
    outside_down[:-1, :] = mask_bin[1:, :] == 0
    outside_left = np.zeros((h, w), dtype=bool)
    outside_left[:, 1:] = mask_bin[:, :-1] == 0
    outside_right = np.zeros((h, w), dtype=bool)
    outside_right[:, :-1] = mask_bin[:, 1:] == 0

    inside_up = np.zeros((h, w), dtype=bool)
    inside_up[1:, :] = mask_bin[:-1, :] == 1
    inside_down = np.zeros((h, w), dtype=bool)
    inside_down[:-1, :] = mask_bin[1:, :] == 1
    inside_left = np.zeros((h, w), dtype=bool)
    inside_left[:, 1:] = mask_bin[:, :-1] == 1
    inside_right = np.zeros((h, w), dtype=bool)
    inside_right[:, :-1] = mask_bin[:, 1:] == 1

    side_specs = {
        "top": {
            "inner_strip": inner_strip & outside_up,
            "outer_strip": outer_strip & inside_down,
            "geom_seed": (inner_edge & outside_up) | (outer_edge & inside_down),
            "sigma_x": effective_sigma,
            "sigma_y": strip_cross_sigma,
        },
        "bottom": {
            "inner_strip": inner_strip & outside_down,
            "outer_strip": outer_strip & inside_up,
            "geom_seed": (inner_edge & outside_down) | (outer_edge & inside_up),
            "sigma_x": effective_sigma,
            "sigma_y": strip_cross_sigma,
        },
        "left": {
            "inner_strip": inner_strip & outside_left,
            "outer_strip": outer_strip & inside_right,
            "geom_seed": (inner_edge & outside_left) | (outer_edge & inside_right),
            "sigma_x": strip_cross_sigma,
            "sigma_y": effective_sigma,
        },
        "right": {
            "inner_strip": inner_strip & outside_right,
            "outer_strip": outer_strip & inside_left,
            "geom_seed": (inner_edge & outside_right) | (outer_edge & inside_left),
            "sigma_x": strip_cross_sigma,
            "sigma_y": effective_sigma,
        },
    }

    side_fields = []
    for spec in side_specs.values():
        if not np.any(spec["geom_seed"]):
            continue

        inner_side = spec["inner_strip"]
        outer_side = spec["outer_strip"]

        inner_pack = np.zeros((h, w, 4), dtype=np.float32)
        inner_pack[inner_side, :3] = lab[inner_side] * pixel_valid[inner_side, None]
        inner_pack[inner_side, 3] = pixel_valid[inner_side]

        outer_pack = np.zeros((h, w, 4), dtype=np.float32)
        outer_pack[outer_side, :3] = lab[outer_side] * pixel_valid[outer_side, None]
        outer_pack[outer_side, 3] = pixel_valid[outer_side]

        inner_pack_s = _gaussian_blur_aniso(inner_pack, spec["sigma_x"], spec["sigma_y"])
        outer_pack_s = _gaussian_blur_aniso(outer_pack, spec["sigma_x"], spec["sigma_y"])

        inner_wt_s = inner_pack_s[:, :, 3]
        outer_wt_s = outer_pack_s[:, :, 3]
        inner_mean = np.where(
            inner_wt_s[:, :, None] > eps,
            inner_pack_s[:, :, :3] / np.maximum(inner_wt_s[:, :, None], eps),
            0.0,
        )
        outer_mean = np.where(
            outer_wt_s[:, :, None] > eps,
            outer_pack_s[:, :, :3] / np.maximum(outer_wt_s[:, :, None], eps),
            0.0,
        )
        side_valid = (inner_wt_s > eps) & (outer_wt_s > eps)
        side_delta = np.where(side_valid[:, :, None], outer_mean - inner_mean, 0.0).astype(np.float32)
        valid_seed = spec["geom_seed"] & side_valid

        geom_dist = _distance_to_seed_map(spec["geom_seed"])
        if np.any(valid_seed):
            valid_dist, indices = distance_transform_edt(
                1 - valid_seed.astype(np.uint8),
                return_indices=True,
            )
            idx_y = indices[0].astype(np.int32)
            idx_x = indices[1].astype(np.int32)
            nearest_delta = side_delta[idx_y, idx_x]
            side_fields.append({
                "geom_dist": geom_dist,
                "valid_dist": valid_dist.astype(np.float32),
                "nearest_delta": nearest_delta,
                "has_valid": True,
            })
        else:
            side_fields.append({
                "geom_dist": geom_dist,
                "valid_dist": np.full((h, w), np.inf, dtype=np.float32),
                "nearest_delta": np.zeros((h, w, 3), dtype=np.float32),
                "has_valid": False,
            })

    if not side_fields:
        return None

    min_geom_dist = np.full((h, w), np.inf, dtype=np.float32)
    for field in side_fields:
        min_geom_dist = np.minimum(min_geom_dist, field["geom_dist"])

    delta_acc = np.zeros((h, w, 3), dtype=np.float32)
    geom_weight_sum = np.zeros((h, w), dtype=np.float32)
    for field in side_fields:
        angle_delta = np.maximum(field["geom_dist"] - min_geom_dist, 0.0)
        geom_weight = _gaussian_distance_weight(angle_delta, corner_mix_sigma)
        geom_weight_sum += geom_weight
        if not field["has_valid"]:
            continue
        validity_gap = np.maximum(field["valid_dist"] - field["geom_dist"], 0.0)
        valid_weight = geom_weight * _gaussian_distance_weight(validity_gap, corner_mix_sigma)
        delta_acc += field["nearest_delta"] * valid_weight[:, :, None]

    return np.where(
        geom_weight_sum[:, :, None] > 1e-8,
        delta_acc / np.maximum(geom_weight_sum[:, :, None], 1e-8),
        0.0,
    ).astype(np.float32)


def _harmonize_by_mask(
    img_u8: np.ndarray,
    mask_np: np.ndarray,
    mode: str = "inside",
    strip_width: int = 8,
    blur_sigma: float = 20.0,
    falloff: int = 64,
    correction_strength: float = 1.0,
    luminance_strength: float = 1.0,
    chroma_strength: float = 1.0,
    mask_threshold: float = 0.5,
    alpha_np=None,
    protect_mask_np=None,
    corner_spread: int = 0,
) -> np.ndarray:
    h, w = img_u8.shape[:2]
    mask_f = np.clip(mask_np.astype(np.float32), 0.0, 1.0)
    mask_bin = (mask_f >= mask_threshold).astype(np.uint8)
    if mask_bin.sum() == 0 or mask_bin.sum() == h * w:
        return img_u8.copy()

    work_scale = min(1.0, MAX_HARMONIZE_WORKDIM / float(max(h, w)))
    if work_scale < 1.0:
        work_h = max(1, int(round(h * work_scale)))
        work_w = max(1, int(round(w * work_scale)))
        work_img_u8 = _resize_array(img_u8, work_w, work_h, cv2.INTER_AREA)
        work_mask = _resize_array(mask_f, work_w, work_h, cv2.INTER_LINEAR).astype(np.float32)
        work_alpha = (
            _resize_array(alpha_np.astype(np.float32), work_w, work_h, cv2.INTER_LINEAR).astype(np.float32)
            if alpha_np is not None else None
        )
        work_protect = (
            _resize_array(protect_mask_np.astype(np.float32), work_w, work_h, cv2.INTER_LINEAR).astype(np.float32)
            if protect_mask_np is not None else None
        )
        delta_smooth = _compute_harmonize_delta(
            work_img_u8,
            work_mask,
            strip_width=max(1, int(round(strip_width * work_scale))),
            blur_sigma=max(float(blur_sigma) * work_scale, 0.5),
            mask_threshold=mask_threshold,
            alpha_np=work_alpha,
            protect_mask_np=work_protect,
            corner_spread=int(round(corner_spread * work_scale)),
        )
        if delta_smooth is None:
            return img_u8.copy()
        delta_smooth = _resize_array(delta_smooth, w, h, cv2.INTER_LINEAR).astype(np.float32)
    else:
        delta_smooth = _compute_harmonize_delta(
            img_u8,
            mask_f,
            strip_width=strip_width,
            blur_sigma=blur_sigma,
            mask_threshold=mask_threshold,
            alpha_np=alpha_np,
            protect_mask_np=protect_mask_np,
            corner_spread=corner_spread,
        )
        if delta_smooth is None:
            return img_u8.copy()

    lab = _to_lab(img_u8)
    alpha_valid = (
        np.clip(alpha_np.astype(np.float32), 0.0, 1.0) if alpha_np is not None
        else np.ones((h, w), dtype=np.float32)
    )
    protect_valid = (
        1.0 - np.clip(protect_mask_np.astype(np.float32), 0.0, 1.0)
        if protect_mask_np is not None
        else np.ones((h, w), dtype=np.float32)
    )
    alpha_f32 = np.clip(alpha_valid * protect_valid, 0.0, 1.0).astype(np.float32)
    channel_scale = np.array([
        correction_strength * luminance_strength,
        correction_strength * chroma_strength,
        correction_strength * chroma_strength,
    ], dtype=np.float32)

    result_lab = lab.copy()
    scale = 0.5 if mode == "both" else 1.0
    if mode in ("inside", "both"):
        dist_in = _fast_distance_transform(mask_bin)
        t_in = np.clip(dist_in / max(float(falloff), 1.0), 0.0, 1.0)
        falloff_in = (0.5 * (1.0 + np.cos(np.pi * t_in))).astype(np.float32)
        weight_in = (falloff_in * mask_f * alpha_f32)[:, :, None] * channel_scale[None, None, :]
        result_lab += delta_smooth * weight_in * scale

    if mode in ("outside", "both"):
        dist_out = _fast_distance_transform(1 - mask_bin)
        t_out = np.clip(dist_out / max(float(falloff), 1.0), 0.0, 1.0)
        falloff_out = (0.5 * (1.0 + np.cos(np.pi * t_out))).astype(np.float32)
        weight_out = (falloff_out * (1.0 - mask_f) * alpha_f32)[:, :, None] * channel_scale[None, None, :]
        result_lab -= delta_smooth * weight_out * scale

    return _lab_to_rgb_u8(result_lab)


def _get_cached_fill_geometry(mask_bool: np.ndarray, cache_key: tuple | None = None) -> dict:
    key = cache_key if cache_key is not None else (
        mask_bool.shape,
        int(mask_bool.sum()),
        bool(mask_bool[0, 0]),
        bool(mask_bool[-1, -1]),
        bool(mask_bool[mask_bool.shape[0] // 2, mask_bool.shape[1] // 2]),
    )
    cached = _FILL_GEOMETRY_CACHE.get(key)
    if cached is not None:
        _FILL_GEOMETRY_CACHE.move_to_end(key)
        return cached

    known_local = ~mask_bool
    known_components, known_labels = cv2.connectedComponents(known_local.astype(np.uint8), connectivity=8)
    dist_to_known, nearest_labels = cv2.distanceTransformWithLabels(
        mask_bool.astype(np.uint8),
        cv2.DIST_L2,
        5,
        labelType=cv2.DIST_LABEL_PIXEL,
    )
    mask_pixels = mask_bool
    components = []
    for comp_id in range(1, known_components):
        comp = known_labels == comp_id
        if not np.any(comp):
            continue
        comp_dist, comp_nearest_labels = cv2.distanceTransformWithLabels(
            (~comp).astype(np.uint8),
            cv2.DIST_L2,
            5,
            labelType=cv2.DIST_LABEL_PIXEL,
        )
        comp_ys, comp_xs = np.where(comp)
        touch_top = int(comp_ys.min()) == 0
        touch_bottom = int(comp_ys.max()) == mask_bool.shape[0] - 1
        touch_left = int(comp_xs.min()) == 0
        touch_right = int(comp_xs.max()) == mask_bool.shape[1] - 1
        corner_touch = (
            (touch_top and touch_left) or
            (touch_top and touch_right) or
            (touch_bottom and touch_left) or
            (touch_bottom and touch_right)
        )
        item = {
            "comp_mask": comp,
            "comp_mask_labels": np.clip(comp_nearest_labels[mask_pixels] - 1, 0, int(comp.sum()) - 1),
            "comp_dist_mask": comp_dist[mask_pixels].astype(np.float32),
            "corner_touch": corner_touch,
        }
        if corner_touch:
            item["comp_nearest_labels_full"] = comp_nearest_labels
        components.append(item)

    cached = {
        "mask_pixels": mask_pixels,
        "known_local": known_local,
        "known_components": known_components,
        "nearest_labels_mask": nearest_labels[mask_pixels],
        "dist_to_known_mask": dist_to_known[mask_pixels].astype(np.float32),
        "nearest_labels_full": nearest_labels,
        "dist_to_known_full": dist_to_known.astype(np.float32),
        "components": components,
    }
    _FILL_GEOMETRY_CACHE[key] = cached
    _FILL_GEOMETRY_CACHE.move_to_end(key)
    while len(_FILL_GEOMETRY_CACHE) > _FILL_GEOMETRY_CACHE_MAX:
        _FILL_GEOMETRY_CACHE.popitem(last=False)
    return cached


def _solve_laplace(bc_grid: np.ndarray) -> np.ndarray:
    h, w = bc_grid.shape
    ni, nj = h - 2, w - 2
    if ni <= 0 or nj <= 0:
        return bc_grid.copy()

    rhs = np.zeros((ni, nj))
    rhs[0, :] -= bc_grid[0, 1:w - 1]
    rhs[-1, :] -= bc_grid[-1, 1:w - 1]
    rhs[:, 0] -= bc_grid[1:h - 1, 0]
    rhs[:, -1] -= bc_grid[1:h - 1, -1]

    ii, jj = np.meshgrid(np.arange(1, ni + 1), np.arange(1, nj + 1), indexing="ij")
    lam = (2 * np.cos(np.pi * ii / (ni + 1)) - 2) + (2 * np.cos(np.pi * jj / (nj + 1)) - 2)
    freq = dstn(rhs, type=1)
    u_interior = dstn(freq / (lam + 1e-12), type=1) / (4 * (ni + 1) * (nj + 1))

    result = bc_grid.copy()
    result[1:-1, 1:-1] = u_interior
    return result


def _prepare_prefill_image_and_mask(
    img_np: np.ndarray,
    fill_mask_np: np.ndarray,
    erase_mask_np: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, bool]:
    effective_mask = np.clip(fill_mask_np.astype(np.float32), 0.0, 1.0)
    if erase_mask_np is None:
        return img_np, effective_mask, False

    erase_mask = np.clip(erase_mask_np.astype(np.float32), 0.0, 1.0)
    effective_mask = np.maximum(effective_mask, erase_mask)
    erase_pixels = erase_mask > 0.1
    if not np.any(erase_pixels):
        return img_np, effective_mask, False

    added_alpha = img_np.shape[2] == 3
    if added_alpha:
        alpha = np.ones((img_np.shape[0], img_np.shape[1], 1), dtype=img_np.dtype)
        prepared = np.concatenate([img_np, alpha], axis=2)
    else:
        prepared = img_np.copy()

    prepared[erase_pixels, :3] = 0.0
    prepared[erase_pixels, 3] = 0.0
    return prepared, effective_mask, added_alpha


def _solve_laplace_with_neumann(
    crop: np.ndarray,
    missing_top: bool,
    missing_bottom: bool,
    missing_left: bool,
    missing_right: bool,
) -> np.ndarray:
    if not (missing_top or missing_bottom or missing_left or missing_right):
        return _solve_laplace(crop)

    h, w = crop.shape
    grid = crop.copy()
    n_anchor = min(8, max(1, min(h, w) // 4))

    def _pick(*vals):
        picked = [float(x) for x in vals if x is not None]
        return float(np.mean(picked)) if picked else None

    tl = _pick(
        float(np.mean(grid[0, 1:1 + n_anchor])) if not missing_top else None,
        float(np.mean(grid[1:1 + n_anchor, 0])) if not missing_left else None,
    )
    tr = _pick(
        float(np.mean(grid[0, -(1 + n_anchor):-1])) if not missing_top else None,
        float(np.mean(grid[1:1 + n_anchor, -1])) if not missing_right else None,
    )
    bl = _pick(
        float(np.mean(grid[-1, 1:1 + n_anchor])) if not missing_bottom else None,
        float(np.mean(grid[-(1 + n_anchor):-1, 0])) if not missing_left else None,
    )
    br = _pick(
        float(np.mean(grid[-1, -(1 + n_anchor):-1])) if not missing_bottom else None,
        float(np.mean(grid[-(1 + n_anchor):-1, -1])) if not missing_right else None,
    )

    known = [v for v in (tl, tr, bl, br) if v is not None]
    fallback = float(np.mean(known)) if known else 0.5
    tl = tl if tl is not None else fallback
    tr = tr if tr is not None else fallback
    bl = bl if bl is not None else fallback
    br = br if br is not None else fallback

    xs = np.linspace(0.0, 1.0, w)
    ys = np.linspace(0.0, 1.0, h)
    if missing_top:
        grid[0, :] = tl * (1.0 - xs) + tr * xs
    if missing_bottom:
        grid[-1, :] = bl * (1.0 - xs) + br * xs
    if missing_left:
        grid[:, 0] = tl * (1.0 - ys) + bl * ys
    if missing_right:
        grid[:, -1] = tr * (1.0 - ys) + br * ys
    return _solve_laplace(grid)


def _fill_components(
    img_np: np.ndarray,
    mask_np: np.ndarray,
    bc_pad: int = 32,
    geometry_cache_key: tuple | None = None,
) -> np.ndarray:
    h, w = mask_np.shape
    channels = img_np.shape[2]
    binary = (mask_np > 0.1).astype(np.int32)
    labeled, n_comp = label(binary)
    filled = img_np.copy()

    def _harmonic_fill_component_global(
        crop_rgb: np.ndarray,
        comp_mask_local: np.ndarray,
        local_cache_key: tuple | None = None,
    ) -> np.ndarray:
        if not np.any(comp_mask_local) or np.all(comp_mask_local):
            return crop_rgb.copy()

        crop_h, crop_w = comp_mask_local.shape
        rgb_channels = min(crop_rgb.shape[2], 3)
        target_max = 96
        if max(crop_h, crop_w) > target_max:
            scale = target_max / float(max(crop_h, crop_w))
            ds_w = max(8, int(round(crop_w * scale)))
            ds_h = max(8, int(round(crop_h * scale)))
        else:
            ds_w, ds_h = crop_w, crop_h

        geometry = _get_cached_fill_geometry(comp_mask_local, local_cache_key)
        known_local = geometry["known_local"]
        if not np.any(known_local):
            return crop_rgb.copy()

        base_rgb = crop_rgb[:, :, :rgb_channels].astype(np.float32)
        known_components = geometry["known_components"]
        mask_pixels = geometry["mask_pixels"]
        known_colors = base_rgb[known_local]
        nearest_rgb_mask = known_colors[
            np.clip(geometry["nearest_labels_mask"] - 1, 0, len(known_colors) - 1)
        ]
        dist_to_known_mask = geometry["dist_to_known_mask"]

        def _build_multi_source_deep_image(local_rgb: np.ndarray, local_geometry: dict) -> np.ndarray:
            local_h, local_w = local_geometry["mask_pixels"].shape
            local_target_max = 96
            if max(local_h, local_w) > local_target_max:
                local_scale = local_target_max / float(max(local_h, local_w))
                local_ds_w = max(8, int(round(local_w * local_scale)))
                local_ds_h = max(8, int(round(local_h * local_scale)))
            else:
                local_ds_w, local_ds_h = local_w, local_h

            local_mask_pixels = local_geometry["mask_pixels"]
            mask_count = int(local_mask_pixels.sum())
            low_accum_rgb = np.zeros((mask_count, rgb_channels), dtype=np.float32)
            low_accum_w = np.zeros((mask_count, 1), dtype=np.float32)
            high_accum_rgb = np.zeros((mask_count, rgb_channels), dtype=np.float32)
            high_accum_w = np.zeros((mask_count, 1), dtype=np.float32)

            for item in local_geometry["components"]:
                comp = item["comp_mask"]
                comp_colors = local_rgb[comp]
                comp_mask_labels = np.clip(item["comp_mask_labels"], 0, len(comp_colors) - 1)
                comp_field_mask_base = comp_colors[comp_mask_labels]

                if item["corner_touch"]:
                    comp_field_full = comp_colors[
                        np.clip(item["comp_nearest_labels_full"] - 1, 0, len(comp_colors) - 1)
                    ]
                    comp_coarse_src = cv2.resize(
                        comp_field_full,
                        (local_ds_w, local_ds_h),
                        interpolation=cv2.INTER_AREA,
                    )
                    comp_coarse = cv2.GaussianBlur(
                        comp_coarse_src,
                        (0, 0),
                        sigmaX=1.6,
                        sigmaY=1.6,
                        borderType=cv2.BORDER_REFLECT_101,
                    )
                    comp_corner_full = cv2.resize(
                        comp_coarse,
                        (local_w, local_h),
                        interpolation=cv2.INTER_LINEAR,
                    ).astype(np.float32)
                    comp_dist_mask = item["comp_dist_mask"]
                    comp_max_dist = float(comp_dist_mask.max()) if comp_dist_mask.size else 1.0
                    corner_radius = max(comp_max_dist * 0.18, 1.0)
                    corner_t_mask = np.clip(comp_dist_mask / corner_radius, 0.0, 1.0)
                    corner_t_mask = corner_t_mask * corner_t_mask * (3.0 - 2.0 * corner_t_mask)
                    comp_field_mask = (
                        comp_field_mask_base * (1.0 - corner_t_mask[:, None]) +
                        comp_corner_full[local_mask_pixels] * corner_t_mask[:, None]
                    )
                    comp_low_coarse = cv2.GaussianBlur(
                        comp_coarse_src,
                        (0, 0),
                        sigmaX=1.8,
                        sigmaY=1.8,
                        borderType=cv2.BORDER_REFLECT_101,
                    )
                    comp_low_mask = cv2.resize(
                        comp_low_coarse,
                        (local_w, local_h),
                        interpolation=cv2.INTER_LINEAR,
                    ).astype(np.float32)[local_mask_pixels]
                else:
                    comp_field_mask = comp_field_mask_base
                    comp_low_mask = comp_field_mask_base

                comp_dist_mask = item["comp_dist_mask"]
                comp_high_mask = comp_field_mask - comp_low_mask
                low_weight = 1.0 / np.power(np.maximum(comp_dist_mask, 1.0), 0.75)
                high_radius = max(float(comp_dist_mask.max()) * 0.12, 24.0) if comp_dist_mask.size else 24.0
                high_weight = (
                    1.0 / np.power(np.maximum(comp_dist_mask, 1.0), 1.1)
                ) * np.exp(-comp_dist_mask / high_radius)

                low_accum_rgb += comp_low_mask * low_weight[:, None]
                low_accum_w += low_weight[:, None]
                high_accum_rgb += comp_high_mask * high_weight[:, None]
                high_accum_w += high_weight[:, None]

            low_mix_mask = low_accum_rgb / np.maximum(low_accum_w, 1e-6)
            high_mix_mask = high_accum_rgb / np.maximum(high_accum_w, 1e-6)
            deep_fill_mask = np.clip(low_mix_mask + high_mix_mask, 0.0, 1.0)
            deep_image = local_rgb.copy()
            deep_image[local_mask_pixels] = deep_fill_mask
            return deep_image

        if known_components > 2:
            blend_scale = 0.25 if max(crop_h, crop_w) >= 512 else 1.0
            if blend_scale < 1.0:
                blend_w = max(8, int(round(crop_w * blend_scale)))
                blend_h = max(8, int(round(crop_h * blend_scale)))
                blend_rgb = cv2.resize(base_rgb, (blend_w, blend_h), interpolation=cv2.INTER_AREA).astype(np.float32)
                blend_mask = cv2.resize(mask_pixels.astype(np.float32), (blend_w, blend_h), interpolation=cv2.INTER_AREA) > 0.05
                blend_cache_key = ("blend_half", blend_h, blend_w, local_cache_key) if local_cache_key is not None else None
                blend_geometry = _get_cached_fill_geometry(blend_mask, blend_cache_key)
                deep_fill_full = cv2.resize(
                    _build_multi_source_deep_image(blend_rgb, blend_geometry),
                    (crop_w, crop_h),
                    interpolation=cv2.INTER_LINEAR,
                ).astype(np.float32)
                deep_fill_mask = deep_fill_full[mask_pixels]
            else:
                deep_fill_mask = _build_multi_source_deep_image(base_rgb, geometry)[mask_pixels]

            max_dist = float(dist_to_known_mask.max()) if dist_to_known_mask.size else 1.0
            blend_radius = max(max_dist * 0.35, 1.0)
            blend_t_mask = np.clip(dist_to_known_mask / blend_radius, 0.0, 1.0)
            blend_t_mask = blend_t_mask * blend_t_mask * (3.0 - 2.0 * blend_t_mask)
            guided_fill_mask = nearest_rgb_mask * (1.0 - blend_t_mask[:, None]) + deep_fill_mask * blend_t_mask[:, None]
            result = crop_rgb.copy()
            result[:, :, :rgb_channels][mask_pixels] = guided_fill_mask
        else:
            nearest_labels = geometry["nearest_labels_full"]
            dist_to_known = geometry["dist_to_known_full"]
            nearest_rgb = known_colors[np.clip(nearest_labels - 1, 0, len(known_colors) - 1)]
            coarse_nearest = cv2.resize(nearest_rgb, (ds_w, ds_h), interpolation=cv2.INTER_AREA)
            coarse_nearest = cv2.GaussianBlur(
                coarse_nearest,
                (0, 0),
                sigmaX=2.0,
                sigmaY=2.0,
                borderType=cv2.BORDER_REFLECT_101,
            )
            upsampled = cv2.resize(coarse_nearest, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
            max_dist = float(dist_to_known[comp_mask_local].max()) if np.any(comp_mask_local) else 1.0
            blend_radius = max(max_dist * 0.55, 1.0)
            blend_t = np.clip(dist_to_known.astype(np.float32) / blend_radius, 0.0, 1.0)
            blend_t = blend_t * blend_t * (3.0 - 2.0 * blend_t)
            guided_fill = nearest_rgb * (1.0 - blend_t[:, :, None]) + upsampled * blend_t[:, :, None]
            result = crop_rgb.copy()
            result[:, :, :rgb_channels][comp_mask_local] = guided_fill[comp_mask_local]

        kernel = np.array([[0.0, 0.25, 0.0], [0.25, 0.0, 0.25], [0.0, 0.25, 0.0]], dtype=np.float32)
        fixed = ~comp_mask_local
        work = result[:, :, :rgb_channels].astype(np.float32, copy=False)
        fixed_values = base_rgb
        avg = cv2.filter2D(work, -1, kernel, borderType=cv2.BORDER_REFLECT_101)
        work[comp_mask_local] = work[comp_mask_local] * 0.7 + avg[comp_mask_local] * 0.3
        work[fixed] = fixed_values[fixed]
        result[:, :, :rgb_channels] = work.astype(np.float64)
        return result

    for k in range(1, n_comp + 1):
        ys, xs = np.where(labeled == k)
        if len(ys) == 0:
            continue

        y0_nat = int(ys.min()) - bc_pad
        y1_nat = int(ys.max()) + bc_pad
        x0_nat = int(xs.min()) - bc_pad
        x1_nat = int(xs.max()) + bc_pad

        missing_top = y0_nat < 0
        missing_bottom = y1_nat >= h
        missing_left = x0_nat < 0
        missing_right = x1_nat >= w

        y0 = max(0, y0_nat)
        y1 = min(h - 1, y1_nat)
        x0 = max(0, x0_nat)
        x1 = min(w - 1, x1_nat)

        comp_local_ys = ys - y0
        comp_local_xs = xs - x0
        comp_mask_local = labeled[y0:y1 + 1, x0:x1 + 1] == k
        use_global_harmonic = missing_top or missing_bottom or missing_left or missing_right

        if use_global_harmonic:
            local_cache_key = geometry_cache_key if (x0 == 0 and y0 == 0 and x1 == w - 1 and y1 == h - 1) else None
            filled_crop_rgb = _harmonic_fill_component_global(
                filled[y0:y1 + 1, x0:x1 + 1, :min(channels, 3)],
                comp_mask_local,
                local_cache_key,
            )
            filled[ys, xs, :min(channels, 3)] = filled_crop_rgb[comp_local_ys, comp_local_xs]
            if channels == 4:
                filled[:, :, 3] = np.where(binary.astype(bool), 1.0, img_np[:, :, 3])
            continue

        for c in range(min(channels, 3)):
            crop = filled[y0:y1 + 1, x0:x1 + 1, c]
            solved = _solve_laplace_with_neumann(
                crop,
                missing_top,
                missing_bottom,
                missing_left,
                missing_right,
            )
            filled[ys, xs, c] = solved[comp_local_ys, comp_local_xs]

    if channels == 4:
        filled[:, :, 3] = np.where(binary.astype(bool), 1.0, img_np[:, :, 3])
    return filled


class PoissonInpaintPrefill:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
            },
            "optional": {
                "erase_mask": ("MASK", {
                    "tooltip": (
                        "Optional mask of pixels to cut out before prefill. "
                        "White pixels are temporarily zeroed to RGB=0, alpha=0, "
                        "then merged into the fill mask and restored by the existing prefill solver."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process"
    CATEGORY = "conditioning/inpaint"

    def process(self, image, mask, erase_mask=None):
        device = image.device
        batch = image.shape[0]
        results = []

        for b in range(batch):
            img_np = image[b].detach().cpu().numpy().astype(np.float64)
            h, w = img_np.shape[:2]

            mask_tensor = mask[b] if mask.ndim == 3 else mask[b, 0]
            geometry_cache_key = None
            if mask_tensor.device.type == "cpu" and mask_tensor.is_contiguous():
                geometry_cache_key = ("torch_mask", tuple(mask_tensor.shape), int(mask_tensor.data_ptr()))

            mask_np = mask_tensor.detach().cpu().numpy().astype(np.float32)
            if mask_np.shape != (h, w):
                mask_np = _resize_array(mask_np, w, h, cv2.INTER_LINEAR).astype(np.float32)
                geometry_cache_key = None

            erase_np = None
            if erase_mask is not None:
                erase_tensor = erase_mask[b] if erase_mask.ndim == 3 else erase_mask[b, 0]
                erase_np = erase_tensor.detach().cpu().numpy().astype(np.float32)
                if erase_np.shape != (h, w):
                    erase_np = _resize_array(erase_np, w, h, cv2.INTER_LINEAR).astype(np.float32)
                geometry_cache_key = None

            prepared_np, effective_mask_np, added_alpha = _prepare_prefill_image_and_mask(
                img_np,
                mask_np,
                erase_np,
            )
            filled_np = _fill_components(
                prepared_np,
                effective_mask_np,
                geometry_cache_key=geometry_cache_key,
            )
            if added_alpha:
                filled_np = filled_np[:, :, :3]
            results.append(torch.from_numpy(filled_np.astype(np.float32)))

        return (torch.stack(results).to(device),)


class MaskHarmonize:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "mode": (["inside", "outside", "both"], {
                    "tooltip": (
                        "inside: correct inside the mask to match the outer region. "
                        "outside: correct outside the mask to match the inner region. "
                        "both: shift both sides toward a common midpoint at the seam."
                    ),
                }),
                "strip_width": ("INT", {"default": 8, "min": 1, "max": 64, "step": 1}),
                "blur_sigma": ("FLOAT", {"default": 20.0, "min": 0.0, "max": 200.0, "step": 1.0}),
                "falloff": ("INT", {"default": 64, "min": 1, "max": 1024, "step": 1}),
                "correction_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "luminance_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "chroma_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "mask_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "corner_spread": ("INT", {"default": 0, "min": 0, "max": 512, "step": 4}),
            },
            "optional": {
                "protect_mask": ("MASK", {
                    "tooltip": (
                        "Optional mask of pixels that must stay untouched and must "
                        "not participate in seam statistics. White = protect."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "harmonize"
    CATEGORY = "image/postprocessing"

    def harmonize(
        self,
        image,
        mask,
        mode,
        strip_width,
        blur_sigma,
        falloff,
        correction_strength,
        luminance_strength,
        chroma_strength,
        mask_threshold,
        corner_spread,
        protect_mask=None,
    ):
        batch = image.shape[0]
        results = []

        for b in range(batch):
            frame = image[b].detach().cpu().numpy()
            h, w = frame.shape[:2]

            if frame.shape[2] == 4:
                alpha_np = np.clip(frame[:, :, 3], 0.0, 1.0).astype(np.float32)
                img_np = (frame[:, :, :3] * 255).clip(0, 255).astype(np.uint8)
            else:
                alpha_np = None
                img_np = (frame * 255).clip(0, 255).astype(np.uint8)

            mask_tensor = mask[b] if mask.ndim == 3 else mask[b, 0]
            mask_np = mask_tensor.detach().cpu().numpy().astype(np.float32)
            if mask_np.shape != (h, w):
                mask_np = _resize_array(mask_np, w, h, cv2.INTER_LINEAR).astype(np.float32)

            protect_np = None
            if protect_mask is not None:
                protect_tensor = protect_mask[b] if protect_mask.ndim == 3 else protect_mask[b, 0]
                protect_np = protect_tensor.detach().cpu().numpy().astype(np.float32)
                if protect_np.shape != (h, w):
                    protect_np = _resize_array(protect_np, w, h, cv2.INTER_LINEAR).astype(np.float32)

            result_np = _harmonize_by_mask(
                img_np,
                mask_np,
                mode=mode,
                strip_width=strip_width,
                blur_sigma=blur_sigma,
                falloff=falloff,
                correction_strength=correction_strength,
                luminance_strength=luminance_strength,
                chroma_strength=chroma_strength,
                mask_threshold=mask_threshold,
                alpha_np=alpha_np,
                protect_mask_np=protect_np,
                corner_spread=corner_spread,
            )
            result_t = torch.from_numpy(result_np.astype(np.float32) / 255.0)
            if alpha_np is not None:
                result_t = torch.cat((result_t, torch.from_numpy(alpha_np).unsqueeze(-1)), dim=-1)
            results.append(result_t)

        return (torch.stack(results),)


NODE_CLASS_MAPPINGS = {
    "PoissonInpaintPrefill": PoissonInpaintPrefill,
    "MaskHarmonize": MaskHarmonize,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PoissonInpaintPrefill": "Poisson Inpaint Prefill",
    "MaskHarmonize": "Mask Harmonize",
}

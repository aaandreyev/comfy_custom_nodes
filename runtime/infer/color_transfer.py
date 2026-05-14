"""
Color transfer algorithms (Reinhard / MKL in Lab, histogram matching), ported from
ComfyUI ``comfy_extras/nodes_post_processing.ColorTransfer``.
Optional region-limited path: bounding box of mask > threshold, transfer on crop, paste back.
"""

from __future__ import annotations

import torch

try:
    import kornia
except ImportError as e:  # pragma: no cover
    kornia = None
    _KORNIA_IMPORT_ERROR = e
else:
    _KORNIA_IMPORT_ERROR = None


def _require_kornia() -> None:
    if kornia is None:
        raise RuntimeError(
            "masked color transfer requires the `kornia` package (install e.g. `pip install kornia`). "
            f"Original import error: {_KORNIA_IMPORT_ERROR}"
        ) from _KORNIA_IMPORT_ERROR


def _bbox_from_binary_mask(mask_hw: torch.Tensor) -> tuple[int, int, int, int] | None:
    """mask_hw: H,W boolean or nonzero pixels."""
    pts = torch.nonzero(mask_hw, as_tuple=False)
    if pts.numel() == 0:
        return None
    y0 = int(pts[:, 0].min().item())
    y1 = int(pts[:, 0].max().item()) + 1
    x0 = int(pts[:, 1].min().item())
    x1 = int(pts[:, 1].max().item()) + 1
    return x0, y0, x1, y1


def _clamp_bbox(x0: int, y0: int, x1: int, y1: int, w: int, h: int) -> tuple[int, int, int, int]:
    x0 = max(0, min(x0, w))
    x1 = max(0, min(x1, w))
    y0 = max(0, min(y0, h))
    y1 = max(0, min(y1, h))
    if x1 <= x0 or y1 <= y0:
        return 0, 0, w, h
    return x0, y0, x1, y1


def _to_lab(images: torch.Tensor, i: int, device: torch.device) -> torch.Tensor:
    _require_kornia()
    return kornia.color.rgb_to_lab(images[i : i + 1].to(device, dtype=torch.float32).permute(0, 3, 1, 2))


def _pool_stats(images: torch.Tensor, device: torch.device, is_reinhard: bool, eps: float):
    N, C = images.shape[0], images.shape[3]
    HW = images.shape[1] * images.shape[2]
    mean = torch.zeros(C, 1, device=device, dtype=torch.float32)
    for i in range(N):
        mean += _to_lab(images, i, device).view(C, -1).mean(dim=-1, keepdim=True)
    mean /= N
    acc = torch.zeros(C, 1 if is_reinhard else C, device=device, dtype=torch.float32)
    for i in range(N):
        centered = _to_lab(images, i, device).view(C, -1) - mean
        if is_reinhard:
            acc += (centered * centered).mean(dim=-1, keepdim=True)
        else:
            acc += centered @ centered.T / HW
    if is_reinhard:
        return mean, torch.sqrt(acc / N).clamp_min_(eps)
    return mean, acc / N


def _frame_stats(lab_flat: torch.Tensor, hw: int, is_reinhard: bool, eps: float):
    mean = lab_flat.mean(dim=-1, keepdim=True)
    if is_reinhard:
        return mean, lab_flat.std(dim=-1, keepdim=True, unbiased=False).clamp_min_(eps)
    centered = lab_flat - mean
    return mean, centered @ centered.T / hw


def _mkl_matrix(cov_s: torch.Tensor, cov_r: torch.Tensor, eps: float) -> torch.Tensor:
    eig_val_s, eig_vec_s = torch.linalg.eigh(cov_s)
    sqrt_val_s = torch.sqrt(eig_val_s.clamp_min(0)).clamp_min_(eps)
    scaled_v = eig_vec_s * sqrt_val_s.unsqueeze(0)
    mid = scaled_v.T @ cov_r @ scaled_v
    eig_val_m, eig_vec_m = torch.linalg.eigh(mid)
    sqrt_m = torch.sqrt(eig_val_m.clamp_min(0))
    inv_sqrt_s = 1.0 / sqrt_val_s
    inv_scaled_v = eig_vec_s * inv_sqrt_s.unsqueeze(0)
    m_half = (eig_vec_m * sqrt_m.unsqueeze(0)) @ eig_vec_m.T
    return inv_scaled_v @ m_half @ inv_scaled_v.T


def _histogram_lut(src: torch.Tensor, ref: torch.Tensor, bins: int = 256) -> torch.Tensor:
    s_bins = (src * (bins - 1)).long().clamp(0, bins - 1)
    r_bins = (ref * (bins - 1)).long().clamp(0, bins - 1)
    s_hist = torch.zeros(src.shape[0], bins, device=src.device, dtype=src.dtype)
    r_hist = torch.zeros(ref.shape[0], bins, device=ref.device, dtype=ref.dtype)
    ones_s = torch.ones_like(src)
    ones_r = torch.ones_like(ref)
    s_hist.scatter_add_(1, s_bins, ones_s)
    r_hist.scatter_add_(1, r_bins, ones_r)
    s_cdf = s_hist.cumsum(1)
    s_cdf = s_cdf / s_cdf[:, -1:]
    r_cdf = r_hist.cumsum(1)
    r_cdf = r_cdf / r_cdf[:, -1:]
    return torch.searchsorted(r_cdf, s_cdf).clamp_max_(bins - 1).float() / (bins - 1)


def _pooled_cdf(images: torch.Tensor, device: torch.device, num_bins: int = 256) -> torch.Tensor:
    c = images.shape[3]
    hist = torch.zeros(c, num_bins, device=device, dtype=torch.float32)
    for i in range(images.shape[0]):
        frame = images[i].to(device, dtype=torch.float32).permute(2, 0, 1).reshape(c, -1)
        bins = (frame * (num_bins - 1)).long().clamp(0, num_bins - 1)
        hist.scatter_add_(1, bins, torch.ones_like(frame))
    cdf = hist.cumsum(1)
    return cdf / cdf[:, -1:]


def _build_histogram_transform(
    image_target: torch.Tensor,
    image_ref: torch.Tensor,
    device: torch.device,
    stats_mode: str,
    target_index: int,
    b: int,
) -> torch.Tensor | None:
    if stats_mode == "per_frame":
        return None
    r_cdf = _pooled_cdf(image_ref, device)
    if stats_mode == "target_frame":
        ti = min(target_index, b - 1)
        s_cdf = _pooled_cdf(image_target[ti : ti + 1], device)
    else:
        s_cdf = _pooled_cdf(image_target, device)
    return torch.searchsorted(r_cdf, s_cdf).clamp_max_(255).float() / 255.0


def _build_lab_transform(
    image_target: torch.Tensor,
    image_ref: torch.Tensor,
    device: torch.device,
    stats_mode: str,
    target_index: int,
    is_reinhard: bool,
):
    _require_kornia()
    eps = 1e-6
    b, h, w, c = image_target.shape
    b_ref = image_ref.shape[0]
    single_ref = b_ref == 1
    hw = h * w
    hw_ref = image_ref.shape[1] * image_ref.shape[2]

    if single_ref or stats_mode in ("uniform", "target_frame"):
        ref_mean, ref_sc = _pool_stats(image_ref, device, is_reinhard, eps)

    if stats_mode in ("uniform", "target_frame"):
        if stats_mode == "target_frame":
            ti = min(target_index, b - 1)
            s_lab = _to_lab(image_target, ti, device).view(c, -1)
            s_mean, s_sc = _frame_stats(s_lab, hw, is_reinhard, eps)
        else:
            s_mean, s_sc = _pool_stats(image_target, device, is_reinhard, eps)

        if is_reinhard:
            scale = ref_sc / s_sc
            offset = ref_mean - scale * s_mean

            def uniform_tf(src_flat: torch.Tensor, **_k):
                return src_flat * scale + offset

            return uniform_tf
        t_mat = _mkl_matrix(s_sc, ref_sc, eps)
        offset = ref_mean - t_mat @ s_mean

        def uniform_mkl(src_flat: torch.Tensor, **_k):
            return t_mat @ src_flat + offset

        return uniform_mkl

    def per_frame_transform(src_flat: torch.Tensor, frame_idx: int):
        s_mean, s_sc = _frame_stats(src_flat, hw, is_reinhard, eps)
        if single_ref:
            r_mean, r_sc = ref_mean, ref_sc
        else:
            ri = min(frame_idx, b_ref - 1)
            r_mean, r_sc = _frame_stats(_to_lab(image_ref, ri, device).view(c, -1), hw_ref, is_reinhard, eps)
        centered = src_flat - s_mean
        if is_reinhard:
            return centered * (r_sc / s_sc) + r_mean
        t_pf = _mkl_matrix(centered @ centered.T / hw, r_sc, eps)
        return t_pf @ centered + r_mean

    return per_frame_transform


def _color_transfer_no_mask(
    image_target: torch.Tensor,
    image_ref: torch.Tensor,
    *,
    method: str,
    stats_mode: str,
    target_index: int,
    strength: float,
    out_dtype: torch.dtype,
    out_device: torch.device,
) -> torch.Tensor:
    """Full-frame transfer; mutates nothing."""
    _require_kornia()
    device = image_target.device
    b, h, w, c = image_target.shape
    b_ref = image_ref.shape[0]
    out = torch.empty(b, h, w, c, device=out_device, dtype=out_dtype)

    if method == "histogram":
        uniform_lut = _build_histogram_transform(image_target, image_ref, device, stats_mode, target_index, b)
        for i in range(b):
            src = image_target[i].to(device, dtype=torch.float32).permute(2, 0, 1)
            src_flat = src.reshape(c, -1)
            if uniform_lut is not None:
                lut = uniform_lut
            else:
                ri = min(i, b_ref - 1)
                ref = image_ref[ri].to(device, dtype=torch.float32).permute(2, 0, 1).reshape(c, -1)
                lut = _histogram_lut(src_flat, ref)
            bin_idx = (src_flat * 255).long().clamp(0, 255)
            matched = lut.gather(1, bin_idx).view(c, h, w)
            result = matched if strength == 1.0 else torch.lerp(src, matched, strength)
            out[i] = result.permute(1, 2, 0).clamp_(0, 1).to(device=out_device, dtype=out_dtype)
    else:
        transform = _build_lab_transform(
            image_target,
            image_ref,
            device,
            stats_mode,
            target_index,
            is_reinhard=method == "reinhard_lab",
        )
        for i in range(b):
            src_frame = _to_lab(image_target, i, device)
            corrected = transform(src_frame.view(c, -1), frame_idx=i)
            if strength == 1.0:
                result = kornia.color.lab_to_rgb(corrected.view(1, c, h, w))
            else:
                result = kornia.color.lab_to_rgb(torch.lerp(src_frame, corrected.view(1, c, h, w), strength))
            out[i] = result.squeeze(0).permute(1, 2, 0).clamp_(0, 1).to(device=out_device, dtype=out_dtype)
    return out


def color_transfer_images(
    image_target: torch.Tensor,
    image_ref: torch.Tensor | None,
    *,
    method: str,
    stats_mode: str,
    target_index: int,
    strength: float,
    mask: torch.Tensor | None = None,
    mask_threshold: float = 0.5,
) -> torch.Tensor:
    """
    Match colors of ``image_target`` to ``image_ref``.

    If ``mask`` is None, transfer applies to the full image.

    If ``mask`` is provided (B,H,W): compute the minimal axis-aligned bbox that contains all
    pixels with ``mask > mask_threshold``. Transfer runs on that rectangular crop (same as a full
    ``ColorTransfer`` on two crops). The crop is merged back with **per-pixel alpha**: only pixels
    under the mask take the transferred color; pixels inside the bbox but outside the mask shape
    stay unchanged — same compositing idea as the RGBA strip in ``extract_mask_neighborhood_strip_rgba``
    (premultiplied RGB × alpha there for viz; here we blend transferred RGB by alpha onto the
    original crop).

    Expects ``image_target`` and ``image_ref`` with identical H×W when ``image_ref`` is set.
    """
    if strength == 0.0 or image_ref is None:
        return image_target.clone()

    if image_target.dim() != 4 or image_ref.dim() != 4:
        raise ValueError("image_target and image_ref must be B×H×W×C tensors")

    out_dtype = image_target.dtype
    out_device = image_target.device
    _, h, w, _ = image_target.shape

    if image_ref.shape[1] != h or image_ref.shape[2] != w:
        raise ValueError(
            f"image_ref spatial size {image_ref.shape[1]}×{image_ref.shape[2]} must match "
            f"image_target {h}×{w}"
        )

    if mask is None:
        return _color_transfer_no_mask(
            image_target,
            image_ref,
            method=method,
            stats_mode=stats_mode,
            target_index=target_index,
            strength=strength,
            out_dtype=out_dtype,
            out_device=out_device,
        )

    if mask.dim() != 3:
        raise ValueError("mask must be B×H×W")

    mb = mask.shape[0]
    if mb not in (1, image_target.shape[0]):
        raise ValueError("mask batch must be 1 or match image_target batch")

    binary_stack = mask > float(mask_threshold)
    out = image_target.clone()
    b = image_target.shape[0]

    for i in range(b):
        mbin = binary_stack[i] if mb > 1 else binary_stack[0]
        bbox = _bbox_from_binary_mask(mbin)
        if bbox is None:
            continue
        x0, y0, x1, y1 = _clamp_bbox(*bbox, w, h)
        if x1 <= x0 or y1 <= y0:
            continue

        crop_t = image_target[i : i + 1, y0:y1, x0:x1, :].contiguous()
        ri = min(i, image_ref.shape[0] - 1)
        crop_r = image_ref[ri : ri + 1, y0:y1, x0:x1, :].contiguous()

        transferred = _color_transfer_no_mask(
            crop_t,
            crop_r,
            method=method,
            stats_mode=stats_mode,
            target_index=min(target_index, max(0, crop_t.shape[0] - 1)),
            strength=strength,
            out_dtype=out_dtype,
            out_device=out_device,
        )

        mask_crop = mask[i, y0:y1, x0:x1] if mb > 1 else mask[0, y0:y1, x0:x1]
        alpha = (mask_crop.float() > float(mask_threshold)).unsqueeze(-1).clamp(0.0, 1.0)

        orig_c = crop_t[0].to(dtype=out_dtype, device=out_device)
        xfer_c = transferred[0].to(dtype=out_dtype, device=out_device)
        composed = orig_c * (1.0 - alpha) + xfer_c * alpha
        out[i, y0:y1, x0:x1, :] = composed

    return out

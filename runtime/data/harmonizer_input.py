from __future__ import annotations

import torch
import torch.nn.functional as F


def rgb_to_luma(strip_rgb: torch.Tensor) -> torch.Tensor:
    weights = strip_rgb.new_tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1)
    return (strip_rgb * weights).sum(dim=1, keepdim=True)


def gradient_magnitude(strip_rgb: torch.Tensor) -> torch.Tensor:
    luma = rgb_to_luma(strip_rgb)
    dx = luma[..., :, 1:] - luma[..., :, :-1]
    dy = luma[..., 1:, :] - luma[..., :-1, :]
    dx = F.pad(dx, (0, 1, 0, 0), mode="replicate")
    dy = F.pad(dy, (0, 0, 0, 1), mode="replicate")
    return torch.sqrt(dx.square() + dy.square() + 1e-6)


def build_harmonizer_input(
    strip_rgb: torch.Tensor,
    *,
    outer_width: int = 128,
    boundary_band_px: int = 24,
    seam_x: int | torch.Tensor | list[int] | tuple[int, ...] | None = None,
) -> dict[str, torch.Tensor]:
    squeeze = False
    if strip_rgb.ndim == 3:
        strip_rgb = strip_rgb.unsqueeze(0)
        squeeze = True
    if strip_rgb.ndim != 4 or strip_rgb.shape[1] != 3:
        raise ValueError("strip_rgb must be CHW or BCHW with 3 RGB channels")
    batch, _, height, width = strip_rgb.shape
    device = strip_rgb.device
    dtype = strip_rgb.dtype
    if seam_x is None:
        seam = torch.full((batch,), float(outer_width), device=device, dtype=dtype)
    elif isinstance(seam_x, torch.Tensor):
        seam = seam_x.to(device=device, dtype=dtype).view(-1)
    else:
        seam = torch.as_tensor(seam_x, device=device, dtype=dtype).view(-1)
    if seam.numel() == 1:
        seam = seam.expand(batch)
    if seam.numel() != batch:
        raise ValueError(f"seam_x batch mismatch: expected {batch}, got {seam.numel()}")
    xs = torch.arange(width, device=device, dtype=dtype).view(1, 1, 1, width)
    seam_v = seam.view(batch, 1, 1, 1)
    inner_width_t = (float(width) - seam).clamp_min(1.0).view(batch, 1, 1, 1)
    mask = (xs >= seam_v).to(dtype).expand(batch, 1, height, width)
    distance = ((xs - seam_v).clamp(min=0.0) / (inner_width_t - 1.0).clamp_min(1.0)).clamp(0.0, 1.0).expand(batch, 1, height, width)
    boundary = (xs.sub(seam_v).abs() <= float(boundary_band_px)).to(dtype).expand(batch, 1, height, width)
    decay_t = ((xs - seam_v) / inner_width_t.clamp_min(1.0)).clamp(0.0, 1.0)
    decay = 0.5 * (1.0 + torch.cos(torch.pi * decay_t))
    decay = torch.where(xs < seam_v, torch.zeros_like(decay), decay).expand(batch, 1, height, width)
    luma = rgb_to_luma(strip_rgb)
    grad = gradient_magnitude(strip_rgb)
    model_input = torch.cat([strip_rgb, mask, distance, boundary, decay, luma, grad], dim=1)
    out = {
        "input": model_input,
        "input_rgb": strip_rgb,
        "mask": mask,
        "distance": distance,
        "boundary_band_mask": boundary,
        "decay_mask": decay,
        "luma": luma,
        "gradient": grad,
        "inner_region_mask": mask,
    }
    if squeeze:
        return {key: value.squeeze(0) for key, value in out.items()}
    return out

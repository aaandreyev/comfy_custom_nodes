from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt


def _resize_mask(mask: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if mask.ndim == 4:
        mask_3d = mask[:, 0].float()
    else:
        mask_3d = mask.float()
    if mask_3d.shape[-2:] == (height, width):
        return mask_3d
    mode = "nearest-exact" if "nearest-exact" in torch._C._nn.__dict__ else "nearest"
    resized = F.interpolate(mask_3d.unsqueeze(1), size=(height, width), mode=mode)
    return resized[:, 0]


def _match_batch(image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if image.shape[0] == mask.shape[0]:
        return image, mask
    if image.shape[0] == 1:
        return image.expand(mask.shape[0], -1, -1, -1).clone(), mask
    if mask.shape[0] == 1:
        return image, mask.expand(image.shape[0], -1, -1).clone()
    raise ValueError(f"Incompatible batch sizes: image={image.shape[0]} mask={mask.shape[0]}")


def extract_mask_neighborhood_pixels(
    image: torch.Tensor,
    mask: torch.Tensor,
    *,
    outer_band_px: int,
    mask_threshold: float = 0.5,
) -> list[torch.Tensor]:
    if image.ndim != 4:
        raise ValueError(f"Expected image BHWC tensor, got {tuple(image.shape)}")
    if mask.ndim not in {3, 4}:
        raise ValueError(f"Expected mask BHW or B1HW tensor, got {tuple(mask.shape)}")
    if outer_band_px <= 0:
        raise ValueError("outer_band_px must be > 0")

    height, width = int(image.shape[1]), int(image.shape[2])
    mask = _resize_mask(mask, height, width)
    image, mask = _match_batch(image, mask)

    outputs: list[torch.Tensor] = []
    for index in range(image.shape[0]):
        mask_np = (mask[index].detach().cpu().numpy() > float(mask_threshold)).astype(np.uint8)
        if int(mask_np.sum()) == 0:
            raise ValueError("Mask has no active pixels")
        outside = mask_np == 0
        dist_to_mask = distance_transform_edt(outside)
        band = outside & (dist_to_mask > 0.0) & (dist_to_mask <= float(outer_band_px))
        if not np.any(band):
            raise ValueError("Mask neighborhood is empty for the requested outer_band_px")
        band_tensor = torch.from_numpy(band).to(device=image.device, dtype=torch.bool)
        pixels = image[index][band_tensor]
        if pixels.numel() == 0:
            raise ValueError("Mask neighborhood produced zero pixels after extraction")
        outputs.append(pixels)
    return outputs


def pack_reference_pixels(
    pixels_per_item: list[torch.Tensor],
    *,
    min_output_size: int,
    max_output_size: int,
) -> torch.Tensor:
    if not pixels_per_item:
        raise ValueError("pixels_per_item must not be empty")
    if min_output_size <= 0 or max_output_size <= 0 or max_output_size < min_output_size:
        raise ValueError("Invalid output size range")

    item_sides: list[int] = []
    for pixels in pixels_per_item:
        pixel_count = int(pixels.shape[0])
        target = int(math.ceil(math.sqrt(pixel_count)))
        item_sides.append(max(min_output_size, min(max_output_size, target)))
    global_side = max(item_sides)
    capacity = global_side * global_side

    packed: list[torch.Tensor] = []
    for pixels in pixels_per_item:
        pixel_count = int(pixels.shape[0])
        if pixel_count >= capacity:
            indices = torch.linspace(0, pixel_count - 1, capacity, device=pixels.device)
            chosen = pixels[indices.round().long()]
        else:
            repeats = int(math.ceil(capacity / max(pixel_count, 1)))
            chosen = pixels.repeat((repeats, 1))[:capacity]
        packed.append(chosen.view(global_side, global_side, pixels.shape[-1]))
    return torch.stack(packed, dim=0)


def build_mask_band_reference_image(
    image: torch.Tensor,
    mask: torch.Tensor,
    *,
    outer_band_px: int,
    min_output_size: int,
    max_output_size: int,
    mask_threshold: float = 0.5,
) -> torch.Tensor:
    pixels = extract_mask_neighborhood_pixels(
        image,
        mask,
        outer_band_px=outer_band_px,
        mask_threshold=mask_threshold,
    )
    return pack_reference_pixels(
        pixels,
        min_output_size=min_output_size,
        max_output_size=max_output_size,
    )

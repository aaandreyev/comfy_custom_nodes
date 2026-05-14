from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
import torch.nn.functional as F


def _round_half_away_from_zero(value: float) -> int:
    if value >= 0:
        return int(math.floor(value + 0.5))
    return -int(math.floor(-value + 0.5))


def _clone_or_none(tensor: torch.Tensor | None) -> torch.Tensor | None:
    return None if tensor is None else tensor.clone()


def _same_hw(tensor: torch.Tensor, width: int, height: int) -> bool:
    return int(tensor.shape[-1]) == int(width) and int(tensor.shape[-2]) == int(height)


def _to_nchw(image: torch.Tensor) -> torch.Tensor:
    return image.permute(0, 3, 1, 2).contiguous()


def _to_nhwc(image: torch.Tensor) -> torch.Tensor:
    return image.permute(0, 2, 3, 1).contiguous()


def _nearest_mode() -> str:
    return "nearest-exact" if "nearest-exact" in torch._C._nn.__dict__ else "nearest"


def _resize_image(image: torch.Tensor, width: int, height: int, algorithm: str) -> torch.Tensor:
    if _same_hw(image, width, height):
        return image
    nchw = _to_nchw(image)
    src_h = int(nchw.shape[-2])
    src_w = int(nchw.shape[-1])
    mode = algorithm.lower()
    if mode == "nearest":
        resized = F.interpolate(nchw, size=(height, width), mode=_nearest_mode())
    elif mode == "bilinear":
        resized = F.interpolate(nchw, size=(height, width), mode="bilinear", align_corners=False, antialias=True)
    elif mode == "bicubic":
        resized = F.interpolate(nchw, size=(height, width), mode="bicubic", align_corners=False, antialias=True)
    elif mode == "box":
        if width <= src_w and height <= src_h:
            resized = F.interpolate(nchw, size=(height, width), mode="area")
        else:
            resized = F.interpolate(nchw, size=(height, width), mode="bilinear", align_corners=False, antialias=True)
    elif mode in {"hamming", "lanczos"}:
        resized = F.interpolate(nchw, size=(height, width), mode="bicubic", align_corners=False, antialias=True)
    else:
        resized = F.interpolate(nchw, size=(height, width), mode="bilinear", align_corners=False, antialias=True)
    return _to_nhwc(resized)


def _resize_mask_nearest(mask: torch.Tensor, width: int, height: int) -> torch.Tensor:
    if _same_hw(mask.unsqueeze(1), width, height):
        return mask
    resized = F.interpolate(mask.unsqueeze(1).float(), size=(height, width), mode=_nearest_mode())
    return resized[:, 0]


def _resize_alpha(mask: torch.Tensor, width: int, height: int) -> torch.Tensor:
    if _same_hw(mask.unsqueeze(1), width, height):
        return mask
    resized = F.interpolate(mask.unsqueeze(1).float(), size=(height, width), mode="bilinear", align_corners=False, antialias=True)
    return resized[:, 0].clamp(0.0, 1.0)


def _match_batch(image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if image.shape[0] == mask.shape[0]:
        return image, mask
    if image.shape[0] == 1:
        return image.expand(mask.shape[0], -1, -1, -1).clone(), mask
    if mask.shape[0] == 1:
        return image, mask.expand(image.shape[0], -1, -1).clone()
    raise AssertionError(f"Incompatible batch sizes: image={image.shape[0]}, mask={mask.shape[0]}")


def _ensure_mask_shape(mask: torch.Tensor | None, image: torch.Tensor) -> torch.Tensor:
    if mask is None:
        return torch.zeros((image.shape[0], image.shape[1], image.shape[2]), device=image.device, dtype=image.dtype)
    mask = mask.clone()
    if mask.shape[1:] != image.shape[1:3]:
        if torch.count_nonzero(mask) == 0:
            return torch.zeros((mask.shape[0], image.shape[1], image.shape[2]), device=image.device, dtype=image.dtype)
        raise AssertionError(
            f"Mask dimensions do not match image dimensions. Expected {image.shape[1:3]}, got {mask.shape[1:]}"
        )
    return mask


def _binary_mask(mask: torch.Tensor, threshold: float) -> torch.Tensor:
    if threshold > 0.0:
        return mask >= threshold
    return mask > 0.0


def _fill_holes_single(mask: torch.Tensor) -> torch.Tensor:
    inv = ~mask
    exterior = torch.zeros_like(inv, dtype=torch.bool)
    if inv.shape[0] == 0 or inv.shape[1] == 0:
        return mask
    exterior[0, :] = inv[0, :]
    exterior[-1, :] = inv[-1, :]
    exterior[:, 0] = inv[:, 0]
    exterior[:, -1] = inv[:, -1]
    exterior = exterior.unsqueeze(0).unsqueeze(0).float()
    inv_f = inv.unsqueeze(0).unsqueeze(0)
    while True:
        grown = F.max_pool2d(exterior, kernel_size=3, stride=1, padding=1) > 0.5
        grown = torch.logical_and(grown, inv_f)
        if torch.equal(grown, exterior > 0.5):
            break
        exterior = grown.float()
    holes = torch.logical_and(inv, ~(exterior[0, 0] > 0.5))
    return torch.logical_or(mask, holes)


def _fill_holes(mask: torch.Tensor) -> torch.Tensor:
    return torch.stack([_fill_holes_single(item) for item in mask], dim=0)


def _dilate_mask(mask: torch.Tensor, pixels: int) -> torch.Tensor:
    if pixels <= 0:
        return mask
    kernel = 2 * int(pixels) + 1
    dilated = F.max_pool2d(mask.unsqueeze(1).float(), kernel_size=kernel, stride=1, padding=pixels) > 0.5
    return dilated[:, 0]


def _gaussian_kernel1d(sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    radius = max(1, int(math.ceil(3.0 * sigma)))
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-(coords * coords) / (2.0 * sigma * sigma))
    return kernel / kernel.sum()


def _inward_blend_mask(mask: torch.Tensor, pixels: int) -> torch.Tensor:
    if pixels <= 0:
        return mask.float()
    sigma = max(float(pixels) / 4.0, 1e-6)
    kernel = _gaussian_kernel1d(sigma, mask.device, torch.float32)
    kernel_x = kernel.view(1, 1, 1, -1)
    kernel_y = kernel.view(1, 1, -1, 1)
    src = mask.unsqueeze(1).float()
    pad = kernel.numel() // 2
    blurred = F.pad(src, (pad, pad, pad, pad), mode="replicate")
    blurred = F.conv2d(blurred, kernel_x)
    blurred = F.conv2d(blurred, kernel_y)
    return (blurred[:, 0] * mask.float()).clamp(0.0, 1.0)


def _bbox_from_mask(mask: torch.Tensor) -> tuple[int, int, int, int] | None:
    points = torch.nonzero(mask, as_tuple=False)
    if points.numel() == 0:
        return None
    y0 = int(points[:, 0].min().item())
    y1 = int(points[:, 0].max().item()) + 1
    x0 = int(points[:, 1].min().item())
    x1 = int(points[:, 1].max().item()) + 1
    return (x0, y0, x1, y1)


def _grow_bbox(bbox: tuple[int, int, int, int], factor: float) -> tuple[int, int, int, int]:
    if factor <= 1.0:
        return bbox
    x0, y0, x1, y1 = bbox
    width = x1 - x0
    height = y1 - y0
    target_w = max(width, int(math.ceil(width * factor)))
    target_h = max(height, int(math.ceil(height * factor)))
    add_w = target_w - width
    add_h = target_h - height
    left = add_w // 2
    right = add_w - left
    top = add_h // 2
    bottom = add_h - top
    return (x0 - left, y0 - top, x1 + right, y1 + bottom)


def _union_bbox(
    bbox_a: tuple[int, int, int, int] | None,
    bbox_b: tuple[int, int, int, int] | None,
) -> tuple[int, int, int, int] | None:
    if bbox_a is None:
        return bbox_b
    if bbox_b is None:
        return bbox_a
    return (
        min(bbox_a[0], bbox_b[0]),
        min(bbox_a[1], bbox_b[1]),
        max(bbox_a[2], bbox_b[2]),
        max(bbox_a[3], bbox_b[3]),
    )


def _fit_bbox_to_aspect(bbox: tuple[int, int, int, int], aspect_ratio: float) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    width = x1 - x0
    height = y1 - y0
    if width <= 0 or height <= 0:
        return bbox
    current_ratio = width / height
    if abs(current_ratio - aspect_ratio) < 1e-12:
        return bbox
    center_x = x0 + width / 2.0
    center_y = y0 + height / 2.0
    if current_ratio < aspect_ratio:
        target_w = int(math.ceil(height * aspect_ratio))
        target_h = height
    else:
        target_w = width
        target_h = int(math.ceil(width / aspect_ratio))
    new_x0 = int(math.floor(center_x - target_w / 2.0))
    new_y0 = int(math.floor(center_y - target_h / 2.0))
    return (new_x0, new_y0, new_x0 + target_w, new_y0 + target_h)


def _pad_image_and_masks(
    image: torch.Tensor,
    selection_mask: torch.Tensor,
    context_mask: torch.Tensor,
    left: int,
    top: int,
    right: int,
    bottom: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    image_nchw = _to_nchw(image)
    image_nchw = F.pad(image_nchw, (left, right, top, bottom), mode="replicate")
    image = _to_nhwc(image_nchw)
    selection_mask = F.pad(selection_mask.unsqueeze(1).float(), (left, right, top, bottom), mode="constant", value=1.0)[:, 0] > 0.5
    context_mask = F.pad(context_mask.unsqueeze(1).float(), (left, right, top, bottom), mode="constant", value=0.0)[:, 0] > 0.5
    return image, selection_mask, context_mask


def _extend_image_and_masks(
    image: torch.Tensor,
    selection_mask: torch.Tensor,
    context_mask: torch.Tensor,
    extend_up_factor: float,
    extend_down_factor: float,
    extend_left_factor: float,
    extend_right_factor: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int, int, int]]:
    height = int(image.shape[1])
    width = int(image.shape[2])
    top_delta = _round_half_away_from_zero(height * (extend_up_factor - 1.0))
    bottom_delta = _round_half_away_from_zero(height * (extend_down_factor - 1.0))
    left_delta = _round_half_away_from_zero(width * (extend_left_factor - 1.0))
    right_delta = _round_half_away_from_zero(width * (extend_right_factor - 1.0))
    new_height = height + top_delta + bottom_delta
    new_width = width + left_delta + right_delta
    if new_height <= 0 or new_width <= 0:
        raise AssertionError(f"Invalid extended size: {new_width}x{new_height}")
    pad_left = max(0, left_delta)
    pad_right = max(0, right_delta)
    pad_top = max(0, top_delta)
    pad_bottom = max(0, bottom_delta)
    if pad_left or pad_right or pad_top or pad_bottom:
        image, selection_mask, context_mask = _pad_image_and_masks(
            image,
            selection_mask,
            context_mask,
            pad_left,
            pad_top,
            pad_right,
            pad_bottom,
        )
    crop_x0 = max(0, -left_delta)
    crop_y0 = max(0, -top_delta)
    crop_x1 = crop_x0 + new_width
    crop_y1 = crop_y0 + new_height
    image = image[:, crop_y0:crop_y1, crop_x0:crop_x1, :]
    selection_mask = selection_mask[:, crop_y0:crop_y1, crop_x0:crop_x1]
    context_mask = context_mask[:, crop_y0:crop_y1, crop_x0:crop_x1]
    orig_x = pad_left - crop_x0
    orig_y = pad_top - crop_y0
    orig_x0 = max(0, orig_x)
    orig_y0 = max(0, orig_y)
    orig_x1 = min(new_width, orig_x + width)
    orig_y1 = min(new_height, orig_y + height)
    orig_box = (orig_x0, orig_y0, max(0, orig_x1 - orig_x0), max(0, orig_y1 - orig_y0))
    return image, selection_mask, context_mask, orig_box


def _pad_output_to_multiple(
    image: torch.Tensor,
    mask: torch.Tensor,
    multiple: int,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int, int, int]]:
    if multiple <= 0:
        return image, mask, (0, 0, int(image.shape[2]), int(image.shape[1]))
    content_h = int(image.shape[1])
    content_w = int(image.shape[2])
    padded_w = int(math.ceil(content_w / multiple) * multiple)
    padded_h = int(math.ceil(content_h / multiple) * multiple)
    pad_w = padded_w - content_w
    pad_h = padded_h - content_h
    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top
    if not (left or right or top or bottom):
        return image, mask, (0, 0, content_w, content_h)
    image_nchw = _to_nchw(image)
    image_nchw = F.pad(image_nchw, (left, right, top, bottom), mode="replicate")
    image = _to_nhwc(image_nchw)
    mask = F.pad(mask.unsqueeze(1).float(), (left, right, top, bottom), mode="constant", value=0.0)[:, 0]
    return image, mask, (left, top, content_w, content_h)


@dataclass
class SingleCropResult:
    canvas_image: torch.Tensor
    canvas_to_orig: tuple[int, int, int, int]
    crop_to_canvas: tuple[int, int, int, int]
    output_content_box: tuple[int, int, int, int]
    cropped_image: torch.Tensor
    cropped_mask: torch.Tensor
    blend_mask: torch.Tensor
    crop_support_canvas: torch.Tensor


def _prepare_single_crop(
    image: torch.Tensor,
    mask: torch.Tensor,
    optional_context_mask: torch.Tensor,
    downscale_algorithm: str,
    upscale_algorithm: str,
    preresize: bool,
    preresize_mode: str,
    preresize_min_width: int,
    preresize_min_height: int,
    preresize_max_width: int,
    preresize_max_height: int,
    extend_for_outpainting: bool,
    extend_up_factor: float,
    extend_down_factor: float,
    extend_left_factor: float,
    extend_right_factor: float,
    mask_hipass_filter: float,
    mask_fill_holes: bool,
    mask_expand_pixels: int,
    mask_invert: bool,
    mask_blend_pixels: int,
    context_from_mask_extend_factor: float,
    output_resize_to_target_size: bool,
    output_target_width: int,
    output_target_height: int,
    output_padding: int,
) -> SingleCropResult:
    orig_box = (0, 0, int(image.shape[2]), int(image.shape[1]))
    if preresize:
        current_width = int(image.shape[2])
        current_height = int(image.shape[1])
        target_width = current_width
        target_height = current_height
        if preresize_mode == "ensure minimum resolution":
            scale = max(preresize_min_width / current_width, preresize_min_height / current_height)
            if scale > 1.0:
                target_width = int(math.ceil(current_width * scale))
                target_height = int(math.ceil(current_height * scale))
        elif preresize_mode == "ensure maximum resolution":
            scale = min(preresize_max_width / current_width, preresize_max_height / current_height)
            if scale < 1.0:
                target_width = max(1, int(math.floor(current_width * scale)))
                target_height = max(1, int(math.floor(current_height * scale)))
        elif preresize_mode == "ensure minimum and maximum resolution":
            if preresize_max_width < preresize_min_width or preresize_max_height < preresize_min_height:
                raise AssertionError("Preresize maximums must be >= minimums")
            min_scale = max(preresize_min_width / current_width, preresize_min_height / current_height)
            max_scale = min(preresize_max_width / current_width, preresize_max_height / current_height)
            if min_scale > 1.0 and max_scale < 1.0:
                raise AssertionError("Cannot satisfy both minimum and maximum resize constraints")
            if min_scale > 1.0:
                target_width = int(math.ceil(current_width * min_scale))
                target_height = int(math.ceil(current_height * min_scale))
            elif max_scale < 1.0:
                target_width = max(1, int(math.floor(current_width * max_scale)))
                target_height = max(1, int(math.floor(current_height * max_scale)))
        if target_width != current_width or target_height != current_height:
            algorithm = upscale_algorithm if target_width >= current_width and target_height >= current_height else downscale_algorithm
            image = _resize_image(image, target_width, target_height, algorithm)
            mask = _resize_mask_nearest(mask, target_width, target_height)
            optional_context_mask = _resize_mask_nearest(optional_context_mask, target_width, target_height)

    selection_mask = _binary_mask(mask, float(mask_hipass_filter))
    context_mask = _binary_mask(optional_context_mask, float(mask_hipass_filter))
    if mask_fill_holes:
        selection_mask = _fill_holes(selection_mask)
    if mask_expand_pixels > 0:
        selection_mask = _dilate_mask(selection_mask, int(mask_expand_pixels))
    if mask_invert:
        selection_mask = ~selection_mask
    if extend_for_outpainting:
        image, selection_mask, context_mask, orig_box = _extend_image_and_masks(
            image,
            selection_mask,
            context_mask,
            extend_up_factor,
            extend_down_factor,
            extend_left_factor,
            extend_right_factor,
        )
    bbox = _bbox_from_mask(selection_mask[0])
    if bbox is None:
        bbox = (0, 0, int(image.shape[2]), int(image.shape[1]))
    if context_from_mask_extend_factor > 1.0:
        bbox = _grow_bbox(bbox, float(context_from_mask_extend_factor))
    bbox = _union_bbox(bbox, _bbox_from_mask(context_mask[0])) or bbox

    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    if output_resize_to_target_size:
        requested_content_w = int(output_target_width)
        requested_content_h = int(output_target_height)
    else:
        requested_content_w = int(width)
        requested_content_h = int(height)
    if requested_content_w <= 0 or requested_content_h <= 0:
        raise AssertionError(f"Invalid requested crop size: {requested_content_w}x{requested_content_h}")

    crop_box = _fit_bbox_to_aspect(bbox, requested_content_w / requested_content_h)
    crop_x0, crop_y0, crop_x1, crop_y1 = crop_box
    pad_left = max(0, -crop_x0)
    pad_top = max(0, -crop_y0)
    pad_right = max(0, crop_x1 - int(image.shape[2]))
    pad_bottom = max(0, crop_y1 - int(image.shape[1]))
    if pad_left or pad_top or pad_right or pad_bottom:
        image, selection_mask, _context_mask = _pad_image_and_masks(
            image,
            selection_mask,
            context_mask,
            pad_left,
            pad_top,
            pad_right,
            pad_bottom,
        )
        orig_box = (orig_box[0] + pad_left, orig_box[1] + pad_top, orig_box[2], orig_box[3])
    canvas_to_orig = orig_box
    crop_to_canvas = (crop_x0 + pad_left, crop_y0 + pad_top, crop_x1 - crop_x0, crop_y1 - crop_y0)

    ctc_x, ctc_y, ctc_w, ctc_h = crop_to_canvas
    content_image = image[:, ctc_y : ctc_y + ctc_h, ctc_x : ctc_x + ctc_w, :]
    content_mask = selection_mask[:, ctc_y : ctc_y + ctc_h, ctc_x : ctc_x + ctc_w].float()
    crop_support_canvas = content_mask.clone()

    if output_resize_to_target_size:
        algorithm = upscale_algorithm if requested_content_w >= ctc_w and requested_content_h >= ctc_h else downscale_algorithm
        content_image = _resize_image(content_image, requested_content_w, requested_content_h, algorithm)
        content_mask = _resize_mask_nearest(content_mask, requested_content_w, requested_content_h)

    cropped_image, cropped_mask, output_content_box = _pad_output_to_multiple(content_image, content_mask, int(output_padding))
    blend_mask = _inward_blend_mask(cropped_mask > 0.5, int(mask_blend_pixels))

    return SingleCropResult(
        canvas_image=image,
        canvas_to_orig=canvas_to_orig,
        crop_to_canvas=crop_to_canvas,
        output_content_box=output_content_box,
        cropped_image=cropped_image,
        cropped_mask=(cropped_mask > 0.5).float(),
        blend_mask=blend_mask,
        crop_support_canvas=crop_support_canvas,
    )


def run_zero_drift_crop(
    *,
    image: torch.Tensor,
    downscale_algorithm: str,
    upscale_algorithm: str,
    preresize: bool,
    preresize_mode: str,
    preresize_min_width: int,
    preresize_min_height: int,
    preresize_max_width: int,
    preresize_max_height: int,
    extend_for_outpainting: bool,
    extend_up_factor: float,
    extend_down_factor: float,
    extend_left_factor: float,
    extend_right_factor: float,
    mask_hipass_filter: float,
    mask_fill_holes: bool,
    mask_expand_pixels: int,
    mask_invert: bool,
    mask_blend_pixels: int,
    context_from_mask_extend_factor: float,
    output_resize_to_target_size: bool,
    output_target_width: int,
    output_target_height: int,
    output_padding: int,
    mask: torch.Tensor | None,
    optional_context_mask: torch.Tensor | None,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    image = image.clone()
    mask = _ensure_mask_shape(_clone_or_none(mask), image)
    optional_context_mask = _ensure_mask_shape(_clone_or_none(optional_context_mask), image)
    image, mask = _match_batch(image, mask)
    image, optional_context_mask = _match_batch(image, optional_context_mask)

    if image.shape[0] > 1 and not output_resize_to_target_size:
        raise AssertionError("output_resize_to_target_size must be enabled for image batches")

    stitcher = {
        "downscale_algorithm": downscale_algorithm,
        "upscale_algorithm": upscale_algorithm,
        "canvas_image": [],
        "canvas_to_orig_x": [],
        "canvas_to_orig_y": [],
        "canvas_to_orig_w": [],
        "canvas_to_orig_h": [],
        "cropped_to_canvas_x": [],
        "cropped_to_canvas_y": [],
        "cropped_to_canvas_w": [],
        "cropped_to_canvas_h": [],
        "output_content_x": [],
        "output_content_y": [],
        "output_content_w": [],
        "output_content_h": [],
        "blend_mask_output": [],
        "crop_support_canvas": [],
    }
    cropped_images: list[torch.Tensor] = []
    cropped_masks: list[torch.Tensor] = []

    for index in range(image.shape[0]):
        result = _prepare_single_crop(
            image=image[index : index + 1],
            mask=mask[index : index + 1],
            optional_context_mask=optional_context_mask[index : index + 1],
            downscale_algorithm=downscale_algorithm,
            upscale_algorithm=upscale_algorithm,
            preresize=preresize,
            preresize_mode=preresize_mode,
            preresize_min_width=preresize_min_width,
            preresize_min_height=preresize_min_height,
            preresize_max_width=preresize_max_width,
            preresize_max_height=preresize_max_height,
            extend_for_outpainting=extend_for_outpainting,
            extend_up_factor=extend_up_factor,
            extend_down_factor=extend_down_factor,
            extend_left_factor=extend_left_factor,
            extend_right_factor=extend_right_factor,
            mask_hipass_filter=mask_hipass_filter,
            mask_fill_holes=mask_fill_holes,
            mask_expand_pixels=mask_expand_pixels,
            mask_invert=mask_invert,
            mask_blend_pixels=mask_blend_pixels,
            context_from_mask_extend_factor=context_from_mask_extend_factor,
            output_resize_to_target_size=output_resize_to_target_size,
            output_target_width=output_target_width,
            output_target_height=output_target_height,
            output_padding=output_padding,
        )
        stitcher["canvas_image"].append(result.canvas_image.cpu())
        stitcher["canvas_to_orig_x"].append(result.canvas_to_orig[0])
        stitcher["canvas_to_orig_y"].append(result.canvas_to_orig[1])
        stitcher["canvas_to_orig_w"].append(result.canvas_to_orig[2])
        stitcher["canvas_to_orig_h"].append(result.canvas_to_orig[3])
        stitcher["cropped_to_canvas_x"].append(result.crop_to_canvas[0])
        stitcher["cropped_to_canvas_y"].append(result.crop_to_canvas[1])
        stitcher["cropped_to_canvas_w"].append(result.crop_to_canvas[2])
        stitcher["cropped_to_canvas_h"].append(result.crop_to_canvas[3])
        stitcher["output_content_x"].append(result.output_content_box[0])
        stitcher["output_content_y"].append(result.output_content_box[1])
        stitcher["output_content_w"].append(result.output_content_box[2])
        stitcher["output_content_h"].append(result.output_content_box[3])
        stitcher["blend_mask_output"].append(result.blend_mask.cpu())
        stitcher["crop_support_canvas"].append(result.crop_support_canvas.cpu())
        cropped_images.append(result.cropped_image[0].cpu())
        cropped_masks.append(result.cropped_mask[0].cpu())

    return stitcher, torch.stack(cropped_images, dim=0), torch.stack(cropped_masks, dim=0)


def stitch_zero_drift_result(
    stitcher: dict[str, Any],
    inpainted_image: torch.Tensor,
) -> torch.Tensor:
    batch = int(inpainted_image.shape[0])
    available = len(stitcher["canvas_image"])
    if available not in {1, batch}:
        raise AssertionError(f"Stitch batch mismatch: stitcher={available}, image={batch}")
    outputs: list[torch.Tensor] = []
    for index in range(batch):
        source_index = 0 if available == 1 else index
        canvas_image = stitcher["canvas_image"][source_index].clone().to(inpainted_image.device)
        ctc_x = int(stitcher["cropped_to_canvas_x"][source_index])
        ctc_y = int(stitcher["cropped_to_canvas_y"][source_index])
        ctc_w = int(stitcher["cropped_to_canvas_w"][source_index])
        ctc_h = int(stitcher["cropped_to_canvas_h"][source_index])
        cto_x = int(stitcher["canvas_to_orig_x"][source_index])
        cto_y = int(stitcher["canvas_to_orig_y"][source_index])
        cto_w = int(stitcher["canvas_to_orig_w"][source_index])
        cto_h = int(stitcher["canvas_to_orig_h"][source_index])
        content_x = int(stitcher["output_content_x"][source_index])
        content_y = int(stitcher["output_content_y"][source_index])
        content_w = int(stitcher["output_content_w"][source_index])
        content_h = int(stitcher["output_content_h"][source_index])
        blend_mask_output = stitcher["blend_mask_output"][source_index].to(inpainted_image.device)
        crop_support_canvas = stitcher["crop_support_canvas"][source_index].to(inpainted_image.device)
        one_image = inpainted_image[index : index + 1]
        input_h = int(one_image.shape[1])
        input_w = int(one_image.shape[2])
        if input_w == content_w and input_h == content_h:
            content_image = one_image
            content_mask = blend_mask_output[:, content_y : content_y + content_h, content_x : content_x + content_w]
        else:
            expected_h = int(blend_mask_output.shape[1])
            expected_w = int(blend_mask_output.shape[2])
            if input_w != expected_w or input_h != expected_h:
                raise AssertionError(
                    f"Inpainted image shape {input_w}x{input_h} does not match expected output {expected_w}x{expected_h}"
                )
            content_image = one_image[:, content_y : content_y + content_h, content_x : content_x + content_w, :]
            content_mask = blend_mask_output[:, content_y : content_y + content_h, content_x : content_x + content_w]
        if int(content_image.shape[2]) != ctc_w or int(content_image.shape[1]) != ctc_h:
            algorithm = stitcher["upscale_algorithm"] if ctc_w >= int(content_image.shape[2]) and ctc_h >= int(content_image.shape[1]) else stitcher["downscale_algorithm"]
            content_image = _resize_image(content_image, ctc_w, ctc_h, algorithm)
            content_mask = _resize_alpha(content_mask, ctc_w, ctc_h)
        else:
            content_mask = content_mask.clamp(0.0, 1.0)
        alpha = (content_mask * crop_support_canvas).unsqueeze(-1)
        canvas_crop = canvas_image[:, ctc_y : ctc_y + ctc_h, ctc_x : ctc_x + ctc_w, :]
        blended = alpha * content_image + (1.0 - alpha) * canvas_crop
        unchanged = torch.all(content_image == canvas_crop, dim=-1, keepdim=True)
        blended = torch.where(unchanged, canvas_crop, blended)
        canvas_image[:, ctc_y : ctc_y + ctc_h, ctc_x : ctc_x + ctc_w, :] = blended
        outputs.append(canvas_image[:, cto_y : cto_y + cto_h, cto_x : cto_x + cto_w, :][0].cpu())
    return torch.stack(outputs, dim=0)

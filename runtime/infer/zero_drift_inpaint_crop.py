from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
import torch.nn.functional as F


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


def _binary_mask_default(mask: torch.Tensor) -> torch.Tensor:
    return mask > 0.0


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


def _clamp_bbox(bbox: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    x0 = max(0, min(int(x0), width))
    x1 = max(0, min(int(x1), width))
    y0 = max(0, min(int(y0), height))
    y1 = max(0, min(int(y1), height))
    return (x0, y0, max(x0, x1), max(y0, y1))


def _nearest_multiple_of_8(x: int) -> int:
    """Nearest positive multiple of 8 for spatial sizes (minimum 8)."""
    x = int(x)
    if x <= 0:
        return 8
    rounded = int(round(x / 8.0)) * 8
    return max(8, rounded)


def _align_crop_outputs_to_multiple_of_8(
    cropped_image: torch.Tensor,
    cropped_mask: torch.Tensor,
    blend_mask: torch.Tensor,
    crop_support_canvas: torch.Tensor,
    *,
    natural_w: int,
    natural_h: int,
    downscale_algorithm: str,
    upscale_algorithm: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Resize crop tensors so H,W are multiples of 8 for VAE-friendly inpainting."""
    tw = _nearest_multiple_of_8(natural_w)
    th = _nearest_multiple_of_8(natural_h)
    if tw == natural_w and th == natural_h:
        return cropped_image, cropped_mask, blend_mask, crop_support_canvas, tw, th
    algo = upscale_algorithm if tw >= natural_w and th >= natural_h else downscale_algorithm
    resized_image = _resize_image(cropped_image, tw, th, algo)
    resized_mask = _resize_mask_nearest(cropped_mask, tw, th)
    resized_mask = (resized_mask > 0.5).float()
    resized_blend = _resize_alpha(blend_mask, tw, th)
    resized_support = _resize_mask_nearest(crop_support_canvas, tw, th).clamp(0.0, 1.0)
    return resized_image, resized_mask, resized_blend, resized_support, tw, th


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
    mask_expand_pixels: int,
    mask_blend_pixels: int,
    context_from_mask_extend_factor: float,
    *,
    align_crop_spatial_multiple_of_8: bool,
) -> SingleCropResult:
    orig_box = (0, 0, int(image.shape[2]), int(image.shape[1]))
    selection_mask = _binary_mask_default(mask)
    context_mask = _binary_mask_default(optional_context_mask)
    if mask_expand_pixels > 0:
        selection_mask = _dilate_mask(selection_mask, int(mask_expand_pixels))
    bbox = _bbox_from_mask(selection_mask[0])
    if bbox is None:
        bbox = (0, 0, int(image.shape[2]), int(image.shape[1]))
    if context_from_mask_extend_factor > 1.0:
        bbox = _grow_bbox(bbox, float(context_from_mask_extend_factor))
    bbox = _union_bbox(bbox, _bbox_from_mask(context_mask[0])) or bbox
    crop_box = _clamp_bbox(bbox, int(image.shape[2]), int(image.shape[1]))
    crop_x0, crop_y0, crop_x1, crop_y1 = crop_box
    canvas_to_orig = orig_box
    crop_to_canvas = (crop_x0, crop_y0, crop_x1 - crop_x0, crop_y1 - crop_y0)

    ctc_x, ctc_y, ctc_w, ctc_h = crop_to_canvas
    content_image = image[:, ctc_y : ctc_y + ctc_h, ctc_x : ctc_x + ctc_w, :]
    content_mask = selection_mask[:, ctc_y : ctc_y + ctc_h, ctc_x : ctc_x + ctc_w].float()
    crop_support_canvas = content_mask.clone()
    cropped_image = content_image
    cropped_mask = content_mask
    blend_mask = _inward_blend_mask(cropped_mask > 0.5, int(mask_blend_pixels))
    cropped_mask_bin = (cropped_mask > 0.5).float()

    out_w, out_h = int(ctc_w), int(ctc_h)
    if align_crop_spatial_multiple_of_8:
        cropped_image, cropped_mask_bin, blend_mask, crop_support_canvas, out_w, out_h = (
            _align_crop_outputs_to_multiple_of_8(
                cropped_image,
                cropped_mask_bin,
                blend_mask,
                crop_support_canvas,
                natural_w=int(ctc_w),
                natural_h=int(ctc_h),
                downscale_algorithm=downscale_algorithm,
                upscale_algorithm=upscale_algorithm,
            )
        )

    output_content_box = (0, 0, out_w, out_h)

    return SingleCropResult(
        canvas_image=image,
        canvas_to_orig=canvas_to_orig,
        crop_to_canvas=crop_to_canvas,
        output_content_box=output_content_box,
        cropped_image=cropped_image,
        cropped_mask=cropped_mask_bin,
        blend_mask=blend_mask,
        crop_support_canvas=crop_support_canvas,
    )


def run_zero_drift_crop(
    *,
    image: torch.Tensor,
    downscale_algorithm: str,
    upscale_algorithm: str,
    mask_expand_pixels: int,
    mask_blend_pixels: int,
    context_from_mask_extend_factor: float,
    mask: torch.Tensor | None,
    optional_context_mask: torch.Tensor | None,
    align_crop_spatial_multiple_of_8: bool = True,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    image = image.clone()
    mask = _ensure_mask_shape(_clone_or_none(mask), image)
    optional_context_mask = _ensure_mask_shape(_clone_or_none(optional_context_mask), image)
    image, mask = _match_batch(image, mask)
    image, optional_context_mask = _match_batch(image, optional_context_mask)

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
            mask_expand_pixels=mask_expand_pixels,
            mask_blend_pixels=mask_blend_pixels,
            context_from_mask_extend_factor=context_from_mask_extend_factor,
            align_crop_spatial_multiple_of_8=align_crop_spatial_multiple_of_8,
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
        mh = int(content_mask.shape[1])
        mw = int(content_mask.shape[2])
        if int(crop_support_canvas.shape[1]) != mh or int(crop_support_canvas.shape[2]) != mw:
            crop_support_canvas = _resize_mask_nearest(crop_support_canvas, mw, mh).clamp(0.0, 1.0)
        alpha = (content_mask * crop_support_canvas).unsqueeze(-1)
        canvas_crop = canvas_image[:, ctc_y : ctc_y + ctc_h, ctc_x : ctc_x + ctc_w, :]
        blended = alpha * content_image + (1.0 - alpha) * canvas_crop
        unchanged = torch.all(content_image == canvas_crop, dim=-1, keepdim=True)
        blended = torch.where(unchanged, canvas_crop, blended)
        canvas_image[:, ctc_y : ctc_y + ctc_h, ctc_x : ctc_x + ctc_w, :] = blended
        outputs.append(canvas_image[:, cto_y : cto_y + cto_h, cto_x : cto_x + cto_w, :][0].cpu())
    return torch.stack(outputs, dim=0)

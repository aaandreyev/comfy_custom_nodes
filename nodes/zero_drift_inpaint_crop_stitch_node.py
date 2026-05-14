from __future__ import annotations

import torch

from ..runtime.infer.zero_drift_inpaint_crop import run_zero_drift_crop, stitch_zero_drift_result

try:
    import nodes as comfy_nodes
except ModuleNotFoundError:  # Optional in bare test environments without ComfyUI.
    comfy_nodes = None


MAX_RESOLUTION = getattr(comfy_nodes, "MAX_RESOLUTION", 16384)


class ZeroDriftInpaintCropNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "downscale_algorithm": (["nearest", "bilinear", "bicubic", "lanczos", "box", "hamming"], {"default": "bilinear"}),
                "upscale_algorithm": (["nearest", "bilinear", "bicubic", "lanczos", "box", "hamming"], {"default": "bicubic"}),
                "preresize": ("BOOLEAN", {"default": False}),
                "preresize_mode": (
                    ["ensure minimum resolution", "ensure maximum resolution", "ensure minimum and maximum resolution"],
                    {"default": "ensure minimum resolution"},
                ),
                "preresize_min_width": ("INT", {"default": 1024, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "preresize_min_height": ("INT", {"default": 1024, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "preresize_max_width": ("INT", {"default": MAX_RESOLUTION, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "preresize_max_height": ("INT", {"default": MAX_RESOLUTION, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "mask_fill_holes": ("BOOLEAN", {"default": True}),
                "mask_expand_pixels": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "mask_invert": ("BOOLEAN", {"default": False}),
                "mask_blend_pixels": ("INT", {"default": 32, "min": 0, "max": 256, "step": 1}),
                "mask_hipass_filter": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "extend_for_outpainting": ("BOOLEAN", {"default": False}),
                "extend_up_factor": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01}),
                "extend_down_factor": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01}),
                "extend_left_factor": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01}),
                "extend_right_factor": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01}),
                "context_from_mask_extend_factor": ("FLOAT", {"default": 1.2, "min": 1.0, "max": 100.0, "step": 0.01}),
                "output_resize_to_target_size": ("BOOLEAN", {"default": True}),
                "output_target_width": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 1}),
                "output_target_height": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 1}),
                "output_padding": (["0", "8", "16", "32", "64", "128", "256", "512"], {"default": "32"}),
                "device_mode": (["cpu (deterministic)", "gpu (same geometry)"], {"default": "gpu (same geometry)"}),
            },
            "optional": {
                "mask": ("MASK",),
                "optional_context_mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("STITCHER", "IMAGE", "MASK")
    RETURN_NAMES = ("stitcher", "cropped_image", "cropped_mask")
    FUNCTION = "inpaint_crop"
    CATEGORY = "inpaint"
    DESCRIPTION = "Pixel-stable crop for inpainting. Geometry comes only from binary masks; output padding never changes crop placement."

    def inpaint_crop(
        self,
        image,
        downscale_algorithm,
        upscale_algorithm,
        preresize,
        preresize_mode,
        preresize_min_width,
        preresize_min_height,
        preresize_max_width,
        preresize_max_height,
        extend_for_outpainting,
        extend_up_factor,
        extend_down_factor,
        extend_left_factor,
        extend_right_factor,
        mask_hipass_filter,
        mask_fill_holes,
        mask_expand_pixels,
        mask_invert,
        mask_blend_pixels,
        context_from_mask_extend_factor,
        output_resize_to_target_size,
        output_target_width,
        output_target_height,
        output_padding,
        device_mode,
        mask=None,
        optional_context_mask=None,
    ):
        del device_mode  # The implementation uses one geometry path on every device.
        stitcher, cropped_image, cropped_mask = run_zero_drift_crop(
            image=image,
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
            output_padding=int(output_padding),
            mask=mask,
            optional_context_mask=optional_context_mask,
        )
        return (stitcher, cropped_image, cropped_mask)


class ZeroDriftInpaintStitchNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stitcher": ("STITCHER",),
                "inpainted_image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "inpaint_stitch"
    CATEGORY = "inpaint"
    DESCRIPTION = "Pixel-perfect stitch for Zero Drift Inpaint Crop. Never resamples when not required and never blends outside the selected mask."

    def inpaint_stitch(self, stitcher, inpainted_image):
        return (stitch_zero_drift_result(stitcher, inpainted_image.clone()),)

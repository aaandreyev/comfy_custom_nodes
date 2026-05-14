from __future__ import annotations

import torch

from ..runtime.infer.zero_drift_inpaint_crop import run_zero_drift_crop, stitch_zero_drift_result

class ZeroDriftInpaintCropNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "downscale_algorithm": (["nearest", "bilinear", "bicubic", "lanczos", "box", "hamming"], {"default": "bilinear"}),
                "upscale_algorithm": (["nearest", "bilinear", "bicubic", "lanczos", "box", "hamming"], {"default": "bicubic"}),
                "mask_expand_pixels": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 1}),
                "mask_blend_pixels": ("INT", {"default": 32, "min": 0, "max": 256, "step": 1}),
                "context_from_mask_extend_factor": ("FLOAT", {"default": 1.2, "min": 1.0, "max": 100.0, "step": 0.01}),
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
    DESCRIPTION = "Pixel-stable crop for inpainting. Returns the natural mask-driven crop with no UI-controlled resize, padding, or geometry toggles."

    def inpaint_crop(
        self,
        image,
        downscale_algorithm,
        upscale_algorithm,
        mask_expand_pixels,
        mask_blend_pixels,
        context_from_mask_extend_factor,
        mask=None,
        optional_context_mask=None,
    ):
        stitcher, cropped_image, cropped_mask = run_zero_drift_crop(
            image=image,
            downscale_algorithm=downscale_algorithm,
            upscale_algorithm=upscale_algorithm,
            mask_expand_pixels=mask_expand_pixels,
            mask_blend_pixels=mask_blend_pixels,
            context_from_mask_extend_factor=context_from_mask_extend_factor,
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

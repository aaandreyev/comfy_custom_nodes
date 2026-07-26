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
                "vae_alignment_multiple_of_8": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "mask": ("MASK",),
                "optional_context_mask": ("MASK",),
                "vae_size_multiple": ("INT", {"default": 16, "min": 1, "max": 128, "step": 1,
                                     "tooltip": "Spatial factor of the VAE in use: 16 for FLUX.2, 8 for SD-family. "
                                                "Crop sizes are aligned to this multiple so VAEEncode does not crop pixels."}),
                "size_bucket_px": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 16,
                                   "tooltip": "0 = off. Otherwise the crop box is grown with real canvas pixels so its "
                                              "width/height land on the {bucket_min_px, +step, ...} grid: torch.compile "
                                              "then sees a small fixed set of shapes instead of recompiling per mask. "
                                              "Use a multiple of 16 (e.g. 128)."}),
                "bucket_min_px": ("INT", {"default": 512, "min": 16, "max": 4096, "step": 16,
                                  "tooltip": "Smallest bucket side when size_bucket_px > 0: shorter crops are grown "
                                             "with real context pixels up to this size."}),
                "bucket_max_px": ("INT", {"default": 1536, "min": 64, "max": 8192, "step": 16,
                                  "tooltip": "Longest bucket side when size_bucket_px > 0: bigger crops are uniformly "
                                             "downscaled so the long side lands here; stitch maps them back."}),
            },
        }

    RETURN_TYPES = ("STITCHER", "IMAGE", "MASK")
    RETURN_NAMES = ("stitcher", "cropped_image", "cropped_mask")
    FUNCTION = "inpaint_crop"
    CATEGORY = "inpaint"
    DESCRIPTION = (
        "Pixel-stable crop for inpainting. When enabled, resizes crop outputs to the nearest "
        "width and height divisible by vae_size_multiple (16 for FLUX.2 VAEs, 8 for SD-family) "
        "for VAE-friendly latent grids; stitch maps back to the original crop rectangle on the canvas."
    )

    def inpaint_crop(
        self,
        image,
        downscale_algorithm,
        upscale_algorithm,
        mask_expand_pixels,
        mask_blend_pixels,
        context_from_mask_extend_factor,
        vae_alignment_multiple_of_8,
        mask=None,
        optional_context_mask=None,
        vae_size_multiple=16,
        size_bucket_px=0,
        bucket_min_px=512,
        bucket_max_px=1536,
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
            align_crop_spatial_multiple_of_8=vae_alignment_multiple_of_8,
            spatial_size_multiple=int(vae_size_multiple),
            size_bucket_px=int(size_bucket_px),
            bucket_min_px=int(bucket_min_px),
            bucket_max_px=int(bucket_max_px),
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

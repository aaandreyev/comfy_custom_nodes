from __future__ import annotations

from ..runtime.infer.mask_band_reference import build_mask_band_reference_image


class ColorTransferRefFromMaskBandNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "outer_band_px": ("INT", {"default": 64, "min": 1, "max": 4096, "step": 1, "tooltip": "How far outside the mask boundary to collect reference pixels."}),
                "min_output_size": ("INT", {"default": 128, "min": 16, "max": 4096, "step": 1}),
                "max_output_size": ("INT", {"default": 1024, "min": 16, "max": 4096, "step": 1}),
                "mask_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("reference_image",)
    FUNCTION = "build"
    CATEGORY = "image/postprocessing"
    DESCRIPTION = "Build a dense reference image for ColorTransfer from pixels in a band around the mask."

    def build(self, image, mask, outer_band_px, min_output_size, max_output_size, mask_threshold):
        reference = build_mask_band_reference_image(
            image,
            mask,
            outer_band_px=int(outer_band_px),
            min_output_size=int(min_output_size),
            max_output_size=int(max_output_size),
            mask_threshold=float(mask_threshold),
        )
        return (reference,)

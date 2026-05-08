from __future__ import annotations

import torch
import torch.nn.functional as F

from ..runtime.infer.neighbor_tone_match import apply_freeform_neighbor_tone_match, write_neighbor_tone_debug


class FreeformNeighborToneMatchNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_image": ("IMAGE",),
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "inner_width": ("INT", {"default": 128, "min": 1, "max": 1024, "step": 1}),
                "outer_band_px": ("INT", {"default": 256, "min": 1, "max": 2048, "step": 1,
                                   "tooltip": "Reference strip width outside the free-form mask used to build the LUT."}),
                "inner_flat_top_px": ("INT", {"default": 48, "min": 0, "max": 1024, "step": 1,
                                      "tooltip": "Flat-top region inside the corrected band before the inward fade starts."}),
                "luma_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
                "chroma_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
                "u_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
                "v_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
                "bins": ("INT", {"default": 32, "min": 4, "max": 128, "step": 1}),
                "correction_mode": (["hybrid", "additive", "multiplicative"], {"default": "hybrid"}),
                "lut_mode": (["3d", "2d_luma_curve"], {"default": "3d"}),
                "color_space": (["srgb", "linear"], {"default": "srgb"}),
                "yuv_matrix": (["bt709", "bt601"], {"default": "bt709"}),
                "delta_smoothing_sigma": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 16.0, "step": 0.25}),
                "debug_previews": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = "seam"

    def run(
        self,
        reference_image,
        image,
        mask,
        inner_width,
        outer_band_px,
        inner_flat_top_px,
        luma_strength,
        chroma_strength,
        u_strength,
        v_strength,
        bins,
        correction_mode,
        lut_mode,
        color_space,
        yuv_matrix,
        delta_smoothing_sigma,
        debug_previews,
    ):
        if reference_image.shape != image.shape:
            raise ValueError("reference_image and image must have the same shape")

        ref_bchw = reference_image.permute(0, 3, 1, 2).contiguous()
        img_bchw = image.permute(0, 3, 1, 2).contiguous()
        ref_rgb = ref_bchw[:, :3]
        img_rgb = img_bchw[:, :3]
        alpha = img_bchw[:, 3:] if img_bchw.shape[1] > 3 else None

        if mask.ndim == 3:
            mask_t = mask.unsqueeze(1).float()
        else:
            mask_t = mask.float()
        if mask_t.shape[-2:] != img_rgb.shape[-2:]:
            mask_t = F.interpolate(mask_t, size=img_rgb.shape[-2:], mode="nearest")

        if (mask_t > 0.5).sum() == 0:
            raise ValueError(
                "FreeformNeighborToneMatchNode: mask has no active pixels (no values above 0.5). "
                "Check that the mask is correctly connected and not empty."
            )

        corrected_rgb, debug = apply_freeform_neighbor_tone_match(
            ref_rgb,
            img_rgb,
            mask_t,
            inner_width=int(inner_width),
            outer_band_px=int(outer_band_px),
            inner_flat_top_px=int(inner_flat_top_px),
            luma_strength=float(luma_strength),
            chroma_strength=float(chroma_strength),
            u_strength=float(u_strength),
            v_strength=float(v_strength),
            bins=int(bins),
            correction_mode=str(correction_mode),
            lut_mode=str(lut_mode),
            color_space=str(color_space),
            yuv_matrix=str(yuv_matrix),
            delta_smoothing_sigma=float(delta_smoothing_sigma),
        )
        if debug_previews:
            write_neighbor_tone_debug(ref_rgb, img_rgb, corrected_rgb, debug)
        corrected = torch.cat([corrected_rgb, alpha], dim=1) if alpha is not None else corrected_rgb
        return (corrected.permute(0, 2, 3, 1).contiguous(),)

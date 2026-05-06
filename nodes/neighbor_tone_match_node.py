from __future__ import annotations

import torch
import torch.nn.functional as F

from ..runtime.infer.neighbor_tone_match import apply_neighbor_tone_match, write_neighbor_tone_debug


class NeighborToneMatchNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_image": ("IMAGE",),
                "drift_source_image": ("IMAGE",),
                "generated_image": ("IMAGE",),
                "mask": ("MASK",),
                "inner_width": ("INT", {"default": 128, "min": 1, "max": 1024, "step": 1}),
                "inner_falloff_px": ("INT", {"default": 48, "min": 0, "max": 1024, "step": 1}),
                "process_left": ("BOOLEAN", {"default": True}),
                "process_right": ("BOOLEAN", {"default": True}),
                "process_top": ("BOOLEAN", {"default": True}),
                "process_bottom": ("BOOLEAN", {"default": True}),
                "luma_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
                "chroma_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
                "bins": ("INT", {"default": 32, "min": 4, "max": 128, "step": 1,
                                  "tooltip": "YUV LUT resolution per channel (bins^3 cells). 24-48 is practical range."}),
                "correction_mode": (["hybrid", "additive", "multiplicative"], {"default": "hybrid",
                                     "tooltip": "hybrid=multiplicative luma + additive chroma (best); additive=pure YUV delta; multiplicative=full ratio"}),
                "color_space": (["linear", "srgb"], {"default": "linear",
                                  "tooltip": "linear=no gamma handling; srgb=linearise before correction, re-encode after"}),
                "corner_px": ("INT", {"default": 0, "min": 0, "max": 512, "step": 1,
                               "tooltip": "Corner taper size in pixels. 0 = auto (derived from mask size and inner_width)."}),
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
        drift_source_image,
        generated_image,
        mask,
        inner_width,
        inner_falloff_px,
        process_left,
        process_right,
        process_top,
        process_bottom,
        luma_strength,
        chroma_strength,
        bins,
        correction_mode,
        color_space,
        corner_px,
        debug_previews,
    ):
        if reference_image.shape != generated_image.shape or reference_image.shape != drift_source_image.shape:
            raise ValueError("reference_image, drift_source_image, and generated_image must have the same shape")

        ref_bchw = reference_image.permute(0, 3, 1, 2).contiguous()
        drift_bchw = drift_source_image.permute(0, 3, 1, 2).contiguous()
        gen_bchw = generated_image.permute(0, 3, 1, 2).contiguous()
        ref_rgb = ref_bchw[:, :3]
        drift_rgb = drift_bchw[:, :3]
        gen_rgb = gen_bchw[:, :3]
        alpha = gen_bchw[:, 3:] if gen_bchw.shape[1] > 3 else None

        if mask.ndim == 3:
            mask_t = mask.unsqueeze(1).float()
        else:
            mask_t = mask.float()
        if mask_t.shape[-2:] != gen_rgb.shape[-2:]:
            mask_t = F.interpolate(mask_t, size=gen_rgb.shape[-2:], mode="nearest")

        if (mask_t > 0.5).sum() == 0:
            raise ValueError(
                "NeighborToneMatchNode: mask has no active pixels (no values above 0.5). "
                "Check that the mask is correctly connected and not empty."
            )

        corner_px_val: float | None = None if int(corner_px) == 0 else float(corner_px)

        corrected_rgb, debug = apply_neighbor_tone_match(
            ref_rgb,
            drift_rgb,
            gen_rgb,
            mask_t,
            inner_width=int(inner_width),
            inner_falloff_px=int(inner_falloff_px),
            process_left=bool(process_left),
            process_right=bool(process_right),
            process_top=bool(process_top),
            process_bottom=bool(process_bottom),
            luma_strength=float(luma_strength),
            chroma_strength=float(chroma_strength),
            bins=int(bins),
            correction_mode=str(correction_mode),
            color_space=str(color_space),
            corner_px=corner_px_val,
        )
        if debug_previews:
            write_neighbor_tone_debug(ref_rgb, gen_rgb, corrected_rgb, debug)
        corrected = torch.cat([corrected_rgb, alpha], dim=1) if alpha is not None else corrected_rgb
        return (corrected.permute(0, 2, 3, 1).contiguous(),)

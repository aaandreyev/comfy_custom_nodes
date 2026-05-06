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
                "outer_band_px": ("INT", {"default": 256, "min": 1, "max": 2048, "step": 1,
                                   "tooltip": "Reference strip width outside the mask used to build the LUT. Can be wider than inner_width for a denser color sample."}),
                "inner_flat_top_px": ("INT", {"default": 48, "min": 0, "max": 1024, "step": 1,
                                      "tooltip": "Flat-top region inside the corrected band before the fade starts. 0 = fade across the whole inner band."}),
                "process_left": ("BOOLEAN", {"default": True}),
                "process_right": ("BOOLEAN", {"default": True}),
                "process_top": ("BOOLEAN", {"default": True}),
                "process_bottom": ("BOOLEAN", {"default": True}),
                "luma_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
                "chroma_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
                "u_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
                "v_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
                "bins": ("INT", {"default": 32, "min": 4, "max": 128, "step": 1,
                                  "tooltip": "YUV LUT resolution per channel (bins^3 cells). 24-48 is practical range."}),
                "correction_mode": (["hybrid", "additive", "multiplicative"], {"default": "hybrid",
                                     "tooltip": "hybrid=multiplicative luma + additive chroma (best); additive=pure YUV delta; multiplicative=full ratio"}),
                "lut_mode": (["3d", "2d_luma_curve"], {"default": "3d",
                             "tooltip": "3d=full YUV LUT; 2d_luma_curve=compact 2D chroma LUT + 1D luma curve."}),
                "color_space": (["srgb", "linear"], {"default": "srgb",
                                  "tooltip": "srgb=linearise before correction, re-encode after; linear=no gamma handling"}),
                "yuv_matrix": (["bt709", "bt601"], {"default": "bt709",
                               "tooltip": "bt709 is the modern default; bt601 is kept for compatibility."}),
                "delta_smoothing_sigma": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 16.0, "step": 0.25,
                                         "tooltip": "Gaussian smoothing for the per-pixel delta inside each corrected band."}),
                "corner_px": ("INT", {"default": -1, "min": -1, "max": 512, "step": 1,
                               "tooltip": "Corner taper size in pixels. -1 = auto, 0 = off."}),
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
        outer_band_px,
        inner_flat_top_px,
        process_left,
        process_right,
        process_top,
        process_bottom,
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

        corner_px_val: float | None = None if int(corner_px) < 0 else float(corner_px)

        corrected_rgb, debug = apply_neighbor_tone_match(
            ref_rgb,
            drift_rgb,
            gen_rgb,
            mask_t,
            inner_width=int(inner_width),
            outer_band_px=int(outer_band_px),
            inner_flat_top_px=int(inner_flat_top_px),
            process_left=bool(process_left),
            process_right=bool(process_right),
            process_top=bool(process_top),
            process_bottom=bool(process_bottom),
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
            corner_px=corner_px_val,
        )
        if debug_previews:
            write_neighbor_tone_debug(ref_rgb, gen_rgb, corrected_rgb, debug)
        corrected = torch.cat([corrected_rgb, alpha], dim=1) if alpha is not None else corrected_rgb
        return (corrected.permute(0, 2, 3, 1).contiguous(),)

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..runtime.infer.neighbor_tone_match import write_neighbor_tone_debug
from ..runtime.infer.seam_profile_tone_match import apply_sided_seam_profile_tone_match


class SidedSeamProfileToneMatchNode:
    """Drop-in replacement for NeighborToneMatch in outpaint/edit workflows.

    Same placement and inputs (reference/image/mask, per-side flags, optional
    topology mask), but the correction is the measured cross-seam tone step
    instead of a donor-band colour LUT.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_image": ("IMAGE",),
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "inner_width": ("INT", {"default": 128, "min": 1, "max": 1024, "step": 1,
                                "tooltip": "Correction fades from 1.0 at the seam to 0 this many px inside."}),
                "inner_flat_top_px": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 1}),
                "seam_inset_px": ("INT", {"default": 0, "min": 0, "max": 256, "step": 1,
                                  "tooltip": "The stitch seam lies this many px inside the supplied mask "
                                             "(set to the GrowMask expand amount if the mask was grown)."}),
                "process_left": ("BOOLEAN", {"default": True}),
                "process_right": ("BOOLEAN", {"default": True}),
                "process_top": ("BOOLEAN", {"default": True}),
                "process_bottom": ("BOOLEAN", {"default": True}),
                "lowpass_sigma": ("FLOAT", {"default": 3.0, "min": 0.5, "max": 16.0, "step": 0.25,
                                  "tooltip": "Spatial scale of the tone sampling on both sides of the seam."}),
                "arc_smooth_px": ("FLOAT", {"default": 12.0, "min": 1.0, "max": 128.0, "step": 0.5,
                                  "tooltip": "Smoothing of the measured step along the seam contour; damps "
                                             "content edges that legitimately cross the seam."}),
                "max_correction": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                                   "tooltip": "Per-channel clamp of the applied YUV delta."}),
                "luma_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
                "chroma_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
                "u_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
                "v_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
                "color_space": (["srgb", "linear"], {"default": "srgb"}),
                "yuv_matrix": (["bt709", "bt601"], {"default": "bt709"}),
                "debug_previews": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "topology_mask": ("MASK",),
            },
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
        inner_flat_top_px,
        seam_inset_px,
        process_left,
        process_right,
        process_top,
        process_bottom,
        lowpass_sigma,
        arc_smooth_px,
        max_correction,
        luma_strength,
        chroma_strength,
        u_strength,
        v_strength,
        color_space,
        yuv_matrix,
        debug_previews,
        topology_mask=None,
    ):
        if (reference_image.shape[1:] != image.shape[1:]
                or reference_image.shape[0] not in (1, image.shape[0])):
            raise ValueError(
                "reference_image must match image in H,W,C and have batch 1 or the same batch"
            )

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
                "SidedSeamProfileToneMatchNode: mask has no active pixels (no values above 0.5). "
                "Check that the mask is correctly connected and not empty."
            )

        corrected_rgb, debug = apply_sided_seam_profile_tone_match(
            ref_rgb,
            img_rgb,
            mask_t,
            topology_mask=topology_mask,
            inner_width=int(inner_width),
            inner_flat_top_px=int(inner_flat_top_px),
            seam_inset_px=int(seam_inset_px),
            process_left=bool(process_left),
            process_right=bool(process_right),
            process_top=bool(process_top),
            process_bottom=bool(process_bottom),
            lowpass_sigma=float(lowpass_sigma),
            arc_smooth_px=float(arc_smooth_px),
            max_correction=float(max_correction),
            luma_strength=float(luma_strength),
            chroma_strength=float(chroma_strength),
            u_strength=float(u_strength),
            v_strength=float(v_strength),
            color_space=str(color_space),
            yuv_matrix=str(yuv_matrix),
        )
        if debug_previews:
            write_neighbor_tone_debug(ref_rgb, img_rgb, corrected_rgb, debug)
        corrected = torch.cat([corrected_rgb, alpha], dim=1) if alpha is not None else corrected_rgb
        return (corrected.permute(0, 2, 3, 1).contiguous(),)

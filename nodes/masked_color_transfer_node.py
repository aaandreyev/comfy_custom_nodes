from __future__ import annotations

import torch

from ..runtime.infer.color_transfer import color_transfer_images


class MaskedColorTransferNode:
    """Color transfer like ComfyUI Color Transfer, with optional mask-limited bbox crop."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_target": ("IMAGE",),
                "method": (
                    ["reinhard_lab", "mkl_lab", "histogram"],
                    {"default": "reinhard_lab"},
                ),
                "source_stats": (
                    ["per_frame", "uniform", "target_frame"],
                    {
                        "default": "per_frame",
                        "tooltip": "per_frame: each frame vs ref frame. uniform: pooled source stats. "
                        "target_frame: baseline from target_index applied to all.",
                    },
                ),
                "target_index": (
                    "INT",
                    {"default": 0, "min": 0, "max": 100000, "step": 1},
                ),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "mask_threshold": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Used only when mask is connected: bbox from pixels with mask > threshold.",
                    },
                ),
            },
            "optional": {
                "image_ref": ("IMAGE",),
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "transfer"
    CATEGORY = "image/postprocessing"
    DESCRIPTION = (
        "Match colors of target to reference (Reinhard/MKL in Lab or histogram). "
        "Without mask: whole image. With mask: bbox encloses bright pixels; transfer runs on that crop, "
        "then blended back with mask as alpha so only masked pixels change (free-form), "
        "like RGB×alpha compositing in mask-band previews."
    )

    def transfer(
        self,
        image_target,
        image_ref=None,
        mask=None,
        method="reinhard_lab",
        source_stats="per_frame",
        target_index=0,
        strength=1.0,
        mask_threshold=0.5,
    ):
        if image_ref is None:
            return (image_target,)

        mask_t: torch.Tensor | None = mask
        if mask_t is not None:
            if mask_t.shape[1:] != image_target.shape[1:3]:
                raise ValueError(
                    f"mask H×W {mask_t.shape[1:]} must match image_target {image_target.shape[1:3]}"
                )

        out = color_transfer_images(
            image_target,
            image_ref,
            method=str(method),
            stats_mode=str(source_stats),
            target_index=int(target_index),
            strength=float(strength),
            mask=mask_t,
            mask_threshold=float(mask_threshold),
        )
        return (out,)

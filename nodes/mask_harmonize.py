from __future__ import annotations

import torch

from ..runtime.legacy_cv import _harmonize_by_mask, _resize_array


class MaskHarmonize:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "mode": (["inside", "outside", "both"], {
                    "tooltip": (
                        "inside: correct inside the mask to match the outer region. "
                        "outside: correct outside the mask to match the inner region. "
                        "both: shift both sides toward a common midpoint at the seam."
                    ),
                }),
                "strip_width": ("INT", {"default": 8, "min": 1, "max": 64, "step": 1}),
                "blur_sigma": ("FLOAT", {"default": 20.0, "min": 0.0, "max": 200.0, "step": 1.0}),
                "falloff": ("INT", {"default": 64, "min": 1, "max": 1024, "step": 1}),
                "correction_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "luminance_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "chroma_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "mask_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "corner_spread": ("INT", {"default": 0, "min": 0, "max": 512, "step": 4}),
            },
            "optional": {
                "protect_mask": ("MASK", {
                    "tooltip": (
                        "Optional mask of pixels that must stay untouched and must "
                        "not participate in seam statistics. White = protect."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "harmonize"
    CATEGORY = "image/postprocessing"

    def harmonize(
        self,
        image,
        mask,
        mode,
        strip_width,
        blur_sigma,
        falloff,
        correction_strength,
        luminance_strength,
        chroma_strength,
        mask_threshold,
        corner_spread,
        protect_mask=None,
    ):
        batch = image.shape[0]
        results = []

        for index in range(batch):
            frame = image[index].detach().cpu().numpy()
            height, width = frame.shape[:2]

            if frame.shape[2] == 4:
                alpha_np = frame[:, :, 3].clip(0.0, 1.0).astype("float32")
                img_np = (frame[:, :, :3] * 255).clip(0, 255).astype("uint8")
            else:
                alpha_np = None
                img_np = (frame * 255).clip(0, 255).astype("uint8")

            mask_tensor = mask[index] if mask.ndim == 3 else mask[index, 0]
            mask_np = mask_tensor.detach().cpu().numpy().astype("float32")
            if mask_np.shape != (height, width):
                mask_np = _resize_array(mask_np, width, height, 1).astype("float32")

            protect_np = None
            if protect_mask is not None:
                protect_tensor = protect_mask[index] if protect_mask.ndim == 3 else protect_mask[index, 0]
                protect_np = protect_tensor.detach().cpu().numpy().astype("float32")
                if protect_np.shape != (height, width):
                    protect_np = _resize_array(protect_np, width, height, 1).astype("float32")

            result_np = _harmonize_by_mask(
                img_np,
                mask_np,
                mode=mode,
                strip_width=strip_width,
                blur_sigma=blur_sigma,
                falloff=falloff,
                correction_strength=correction_strength,
                luminance_strength=luminance_strength,
                chroma_strength=chroma_strength,
                mask_threshold=mask_threshold,
                alpha_np=alpha_np,
                protect_mask_np=protect_np,
                corner_spread=corner_spread,
            )
            result_t = torch.from_numpy(result_np.astype("float32") / 255.0)
            if alpha_np is not None:
                result_t = torch.cat((result_t, torch.from_numpy(alpha_np).unsqueeze(-1)), dim=-1)
            results.append(result_t)

        return (torch.stack(results),)

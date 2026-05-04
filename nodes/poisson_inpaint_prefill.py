from __future__ import annotations

import torch

from ..runtime.legacy_cv import (
    _fill_components,
    _prepare_prefill_image_and_mask,
    _resize_array,
)


class PoissonInpaintPrefill:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
            },
            "optional": {
                "erase_mask": ("MASK", {
                    "tooltip": (
                        "Optional mask of pixels to cut out before prefill. "
                        "White pixels are temporarily zeroed to RGB=0, alpha=0, "
                        "then merged into the fill mask and restored by the existing prefill solver."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process"
    CATEGORY = "conditioning/inpaint"

    def process(self, image, mask, erase_mask=None):
        device = image.device
        batch = image.shape[0]
        results = []

        for index in range(batch):
            img_np = image[index].detach().cpu().numpy().astype("float64")
            height, width = img_np.shape[:2]

            mask_tensor = mask[index] if mask.ndim == 3 else mask[index, 0]
            geometry_cache_key = None
            if mask_tensor.device.type == "cpu" and mask_tensor.is_contiguous():
                geometry_cache_key = ("torch_mask", tuple(mask_tensor.shape), int(mask_tensor.data_ptr()))

            mask_np = mask_tensor.detach().cpu().numpy().astype("float32")
            if mask_np.shape != (height, width):
                mask_np = _resize_array(mask_np, width, height, 1).astype("float32")
                geometry_cache_key = None

            erase_np = None
            if erase_mask is not None:
                erase_tensor = erase_mask[index] if erase_mask.ndim == 3 else erase_mask[index, 0]
                erase_np = erase_tensor.detach().cpu().numpy().astype("float32")
                if erase_np.shape != (height, width):
                    erase_np = _resize_array(erase_np, width, height, 1).astype("float32")
                geometry_cache_key = None

            prepared_np, effective_mask_np, added_alpha = _prepare_prefill_image_and_mask(
                img_np,
                mask_np,
                erase_np,
            )
            filled_np = _fill_components(
                prepared_np,
                effective_mask_np,
                geometry_cache_key=geometry_cache_key,
            )
            if added_alpha:
                filled_np = filled_np[:, :, :3]
            results.append(torch.from_numpy(filled_np.astype("float32")))

        return (torch.stack(results).to(device),)

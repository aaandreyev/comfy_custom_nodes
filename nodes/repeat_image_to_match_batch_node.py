"""Repeat an image to the batch size of a reference latent.

`edit_api_v2` and `outpainting_api_v2` finish with `ImageCompositeMasked`, whose destination is
the source image straight off `LoadImage` — batch 1 — while its source carries the N generated
variants. ComfyUI's `composite()` sizes the output by the destination:

    source = comfy.utils.repeat_to_batch_size(source, destination.shape[0])

so with `RepeatLatentBatch.amount = 3` the graph silently returns one image instead of three.
`inpaint_api_v2` is unaffected: it has no composite step, the stitch output goes straight to save.

The obvious patch is `RepeatImageBatch` on the destination, but that puts the batch size in two
places and the caller has to keep them in sync — `comfyui_gateway` sets `amount = len(seeds)` in
one spot today. Taking the count from the latent instead keeps a single source of truth, so a
change to the candidate count cannot desynchronise the graph.
"""

from __future__ import annotations

import torch


class RepeatImageToMatchBatchNode:
    CATEGORY = "prefill_harmonization"
    FUNCTION = "apply"
    RETURN_TYPES = ("IMAGE",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "reference": ("LATENT", {
                    "tooltip": "Batch size is read from this latent, so the graph has one source "
                               "of truth for the candidate count."}),
            }
        }

    def apply(self, image: torch.Tensor, reference: dict):
        samples = reference.get("samples") if isinstance(reference, dict) else None
        if samples is None or image.shape[0] >= samples.shape[0]:
            return (image,)
        repeats = -(-samples.shape[0] // image.shape[0])   # ceil, then trim
        return (image.repeat(repeats, 1, 1, 1)[:samples.shape[0]],)


NODE_CLASS_MAPPINGS = {"RepeatImageToMatchBatch": RepeatImageToMatchBatchNode}
NODE_DISPLAY_NAME_MAPPINGS = {"RepeatImageToMatchBatch": "Repeat Image To Match Batch"}

"""
:class:`Flux2ActiveRegionRoPEOffset` — FCG (Frozen-Context Generation) coordinate alignment.

The FCG inpaint/edit recipe samples ONLY the cropped active region (the mask bbox) while the full
canvas is attached as a ReferenceLatent. RoPE attention depends only on coordinate *differences*,
so for the crop to line up with its reference the active tokens must sit at their true canvas
coordinates. This node shifts the main image's (H, W) RoPE ids by the crop origin; reference
tokens keep their own coordinates (they already start at the canvas origin).

Works with both FLUX.2 backends:

* **ComfyUI-native flux2** (``comfy.ldm.flux.model.Flux``): patches ``diffusion_model.process_img``
  through the ModelPatcher object-patch mechanism. Only calls with ``index == 0`` (the main image)
  are shifted; references arrive with ``index >= 1`` (klein uses ``ref_index_scale=10``) and pass
  through untouched. Note: the rarely-used ``uxo`` ref method also calls with index 0 and would be
  shifted — klein workflows use the default "index" method, so this does not apply.
* **Nunchaku fork wrapper** (``ComfyFlux2Wrapper``): sets
  ``transformer_options["flux2_active_offset"]`` which the wrapper's ``process_img`` consumes
  (see cn-flux2-work ``wrappers/flux2.py``).

Offsets are given in IMAGE pixels of the canvas (what the crop node reports) and converted to
latent-token units with the FLUX.2 factor of 16 (VAE 16x, patch_size 1). Offsets must be multiples
of 16 for exact alignment — the node floors and warns otherwise.

GPU-VERIFY: with kv weights + full-canvas reference, an FCG inpaint with this node should show
strictly better seam metrics (shared_evaluator group A/B) than the same graph without it.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_LATENT_FACTOR = 16  # FLUX.2: VAE 16x spatial, patch_size 1 -> one token per 16 image px


def _token_offset(px: int) -> int:
    if px % _LATENT_FACTOR != 0:
        logger.warning(
            "flux2 active-region offset %d px is not a multiple of %d; flooring — expect a sub-token misalignment.",
            px, _LATENT_FACTOR,
        )
    return px // _LATENT_FACTOR


def make_shifted_process_img(orig_process_img, dy_tokens: int, dx_tokens: int):
    """Wrap a flux2 ``process_img`` so the main image (index==0) is shifted by (dy, dx) tokens."""

    def patched(x, index=0, h_offset=0, w_offset=0, **kwargs):
        if index == 0:
            h_offset = h_offset + dy_tokens
            w_offset = w_offset + dx_tokens
        return orig_process_img(x, index=index, h_offset=h_offset, w_offset=w_offset, **kwargs)

    return patched


class Flux2ActiveRegionRoPEOffset:
    """Shift the sampled (active) latent's RoPE coordinates to its true canvas position."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "FLUX.2 klein model (native or Nunchaku FLUX.2 DiT Loader)."}),
                "x_offset_px": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 16,
                                        "tooltip": "Canvas X of the crop origin, image pixels."}),
                "y_offset_px": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 16,
                                        "tooltip": "Canvas Y of the crop origin, image pixels."}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "flux2/fcg"
    TITLE = "Flux2 Active Region RoPE Offset (FCG)"

    def apply(self, model, x_offset_px: int, y_offset_px: int):
        dy = _token_offset(int(y_offset_px))
        dx = _token_offset(int(x_offset_px))
        if dy == 0 and dx == 0:
            return (model,)

        m = model.clone()
        transformer_options = m.model_options.setdefault("transformer_options", {})
        transformer_options["flux2_active_offset"] = (dy, dx)

        dm = m.model.diffusion_model
        if hasattr(dm, "ctx_for_copy"):
            # Nunchaku fork wrapper: reads flux2_active_offset from transformer_options itself.
            logger.info("FCG rope offset (%d, %d) tokens via transformer_options (nunchaku wrapper).", dy, dx)
        elif hasattr(dm, "process_img"):
            m.add_object_patch("diffusion_model.process_img", make_shifted_process_img(dm.process_img, dy, dx))
            logger.info("FCG rope offset (%d, %d) tokens via process_img object patch (native flux2).", dy, dx)
        else:
            raise ValueError("Model has no flux2 process_img — is this a FLUX.2 model?")
        return (m,)

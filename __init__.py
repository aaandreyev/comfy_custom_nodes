from __future__ import annotations

# Must run before anything compiles. ComfyUI executes prompts on a worker thread and dynamo keeps
# its overrides in a ContextVar, so the settings have to reach the entry's default rather than the
# importing context — see runtime/dynamo_config.py for the measurement.
from .runtime import dynamo_config

dynamo_config.apply()

from .nodes.freeform_neighbor_tone_match_node import FreeformNeighborToneMatchNode
from .nodes.mask_harmonize import MaskHarmonize
from .nodes.neighbor_tone_match_node import NeighborToneMatchNode
from .nodes.poisson_inpaint_prefill import PoissonInpaintPrefill
from .nodes.seam_latent_anchor_node import SeamLatentAnchorNode
from .nodes.seam_harmonizer_node import SeamHarmonizerV3Node
from .nodes.draw_mask_overlay_advanced_node import DrawMaskOverlayAdvancedNode
from .nodes.color_transfer_ref_from_mask_band_node import ColorTransferRefFromMaskBandNode
from .nodes.zero_drift_inpaint_crop_stitch_node import ZeroDriftInpaintCropNode, ZeroDriftInpaintStitchNode
from .nodes.masked_color_transfer_node import MaskedColorTransferNode
from .nodes.flux2_active_region_rope_offset_node import Flux2ActiveRegionRoPEOffset
from .nodes.flux2_compile_nodes import Flux2CLIPCompile, NunchakuFlux2ModelCompile
from .nodes.flux2_nunchaku_te_loader_node import NunchakuQwen3TELoader
from .nodes.seamfix_clip_text_encode_node import SeamfixCLIPTextEncode
from .nodes.seam_profile_tone_match_node import SeamProfileToneMatchNode
from .nodes.sided_seam_profile_tone_match_node import SidedSeamProfileToneMatchNode
from .nodes import SeamGuidedKSamplerNode

try:
    from .nodes.flux2_klein_spatial_denoise_ksampler_node import Flux2KleinSpatialDenoiseKSamplerNode
except ModuleNotFoundError:  # Optional in bare test environments without ComfyUI.
    Flux2KleinSpatialDenoiseKSamplerNode = None


NODE_CLASS_MAPPINGS = {
    "PoissonInpaintPrefill": PoissonInpaintPrefill,
    "MaskHarmonize": MaskHarmonize,
    "FreeformNeighborToneMatch": FreeformNeighborToneMatchNode,
    "NeighborToneMatch": NeighborToneMatchNode,
    "SeamLatentAnchor": SeamLatentAnchorNode,
    "SeamHarmonizerV3": SeamHarmonizerV3Node,
    "DrawMaskOverlayAdvanced": DrawMaskOverlayAdvancedNode,
    "ColorTransferRefFromMaskBand": ColorTransferRefFromMaskBandNode,
    "ZeroDriftInpaintCrop": ZeroDriftInpaintCropNode,
    "ZeroDriftInpaintStitch": ZeroDriftInpaintStitchNode,
    "MaskedColorTransfer": MaskedColorTransferNode,
    "Flux2ActiveRegionRoPEOffset": Flux2ActiveRegionRoPEOffset,
    "Flux2CLIPCompile": Flux2CLIPCompile,
    "NunchakuFlux2ModelCompile": NunchakuFlux2ModelCompile,
    "NunchakuQwen3TELoader": NunchakuQwen3TELoader,
    "SeamfixCLIPTextEncode": SeamfixCLIPTextEncode,
    "SeamProfileToneMatch": SeamProfileToneMatchNode,
    "SidedSeamProfileToneMatch": SidedSeamProfileToneMatchNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PoissonInpaintPrefill": "Poisson Inpaint Prefill",
    "MaskHarmonize": "Mask Harmonize",
    "FreeformNeighborToneMatch": "Freeform Neighbor Tone Match",
    "NeighborToneMatch": "Neighbor Tone Match",
    "SeamLatentAnchor": "Seam Latent Anchor",
    "SeamHarmonizerV3": "Seam Harmonizer v3",
    "DrawMaskOverlayAdvanced": "Draw Mask Overlay Advanced",
    "ColorTransferRefFromMaskBand": "Color Transfer Ref From Mask Band",
    "ZeroDriftInpaintCrop": "Zero Drift Inpaint Crop",
    "ZeroDriftInpaintStitch": "Zero Drift Inpaint Stitch",
    "MaskedColorTransfer": "Masked Color Transfer",
    "Flux2ActiveRegionRoPEOffset": "Flux2 Active Region RoPE Offset (FCG)",
    "Flux2CLIPCompile": "Flux2 CLIP Compile",
    "NunchakuFlux2ModelCompile": "Nunchaku FLUX.2 Model Compile (per-block)",
    "NunchakuQwen3TELoader": "Nunchaku Qwen3 Text Encoder Loader (FLUX.2 klein)",
    "SeamfixCLIPTextEncode": "CLIP Text Encode (SEAMFIX / Inpaint Prompt)",
    "SeamProfileToneMatch": "Seam Profile Tone Match",
    "SidedSeamProfileToneMatch": "Seam Profile Tone Match (Sided)",
}

if Flux2KleinSpatialDenoiseKSamplerNode is not None:
    NODE_CLASS_MAPPINGS["Flux2KleinSpatialDenoiseKSampler"] = Flux2KleinSpatialDenoiseKSamplerNode
    NODE_DISPLAY_NAME_MAPPINGS["Flux2KleinSpatialDenoiseKSampler"] = "Flux2 Klein Spatial Denoise KSampler"

if SeamGuidedKSamplerNode is not None:
    NODE_CLASS_MAPPINGS["SeamGuidedKSampler"] = SeamGuidedKSamplerNode
    NODE_DISPLAY_NAME_MAPPINGS["SeamGuidedKSampler"] = "Seam Guided KSampler"


import os as _os

if _os.environ.get("COMFY_MEGA_CACHE"):
    try:
        from .runtime.compile_cache import load_mega_cache as _load_mega_cache

        print(
            "[prefill-harmonization] mega-cache:",
            _load_mega_cache(_os.environ["COMFY_MEGA_CACHE"]),
        )
    except Exception as _exc:  # noqa: BLE001 - never block node registration
        print("[prefill-harmonization] mega-cache load failed:", _exc)

try:
    from aiohttp import web as _web
    from server import PromptServer as _PromptServer

    from .runtime.compile_cache import save_mega_cache as _save_mega_cache

    @_PromptServer.instance.routes.post("/prefill_harmonization/save_mega_cache")
    async def _save_mega_cache_route(request):
        try:
            data = await request.json()
        except Exception:  # noqa: BLE001 - empty body is fine
            data = {}
        path = data.get("path") or _os.environ.get("COMFY_MEGA_CACHE_OUT") or "/tmp/mega_cache.bin"
        return _web.json_response(_save_mega_cache(str(path)))
except Exception:  # Bare test environments without a running ComfyUI server.
    pass

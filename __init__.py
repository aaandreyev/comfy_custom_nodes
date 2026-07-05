from __future__ import annotations

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
}

if Flux2KleinSpatialDenoiseKSamplerNode is not None:
    NODE_CLASS_MAPPINGS["Flux2KleinSpatialDenoiseKSampler"] = Flux2KleinSpatialDenoiseKSamplerNode
    NODE_DISPLAY_NAME_MAPPINGS["Flux2KleinSpatialDenoiseKSampler"] = "Flux2 Klein Spatial Denoise KSampler"

if SeamGuidedKSamplerNode is not None:
    NODE_CLASS_MAPPINGS["SeamGuidedKSampler"] = SeamGuidedKSamplerNode
    NODE_DISPLAY_NAME_MAPPINGS["SeamGuidedKSampler"] = "Seam Guided KSampler"

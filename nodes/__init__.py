from .freeform_neighbor_tone_match_node import FreeformNeighborToneMatchNode
from .mask_harmonize import MaskHarmonize
from .neighbor_tone_match_node import NeighborToneMatchNode
from .poisson_inpaint_prefill import PoissonInpaintPrefill
from .seam_latent_anchor_node import SeamLatentAnchorNode
from .seam_harmonizer_node import SeamHarmonizerV3Node
from .draw_mask_overlay_advanced_node import DrawMaskOverlayAdvancedNode
from .color_transfer_ref_from_mask_band_node import ColorTransferRefFromMaskBandNode
from .zero_drift_inpaint_crop_stitch_node import ZeroDriftInpaintCropNode, ZeroDriftInpaintStitchNode
from .masked_color_transfer_node import MaskedColorTransferNode
from .flux2_active_region_rope_offset_node import Flux2ActiveRegionRoPEOffset
from .flux2_compile_nodes import Flux2CLIPCompile, NunchakuFlux2ModelCompile
from .flux2_nunchaku_te_loader_node import NunchakuQwen3TELoader
from .seamfix_clip_text_encode_node import SeamfixCLIPTextEncode

try:
    from .flux2_klein_spatial_denoise_ksampler_node import Flux2KleinSpatialDenoiseKSamplerNode
except ModuleNotFoundError:  # Optional in bare test environments without ComfyUI.
    Flux2KleinSpatialDenoiseKSamplerNode = None

try:
    from .seam_guided_ksampler_node import SeamGuidedKSamplerNode
except ModuleNotFoundError:  # Optional in bare test environments without ComfyUI.
    SeamGuidedKSamplerNode = None

__all__ = [
    "MaskHarmonize",
    "FreeformNeighborToneMatchNode",
    "NeighborToneMatchNode",
    "PoissonInpaintPrefill",
    "SeamLatentAnchorNode",
    "SeamHarmonizerV3Node",
    "DrawMaskOverlayAdvancedNode",
    "ColorTransferRefFromMaskBandNode",
    "ZeroDriftInpaintCropNode",
    "ZeroDriftInpaintStitchNode",
    "MaskedColorTransferNode",
    "SeamfixCLIPTextEncode",
]

if Flux2KleinSpatialDenoiseKSamplerNode is not None:
    __all__.append("Flux2KleinSpatialDenoiseKSamplerNode")

if SeamGuidedKSamplerNode is not None:
    __all__.append("SeamGuidedKSamplerNode")

from __future__ import annotations

from .nodes.freeform_neighbor_tone_match_node import FreeformNeighborToneMatchNode
from .nodes.mask_harmonize import MaskHarmonize
from .nodes.neighbor_tone_match_node import NeighborToneMatchNode
from .nodes.poisson_inpaint_prefill import PoissonInpaintPrefill
from .nodes.seam_latent_anchor_node import SeamLatentAnchorNode
from .nodes.seam_harmonizer_node import SeamHarmonizerV3Node
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
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PoissonInpaintPrefill": "Poisson Inpaint Prefill",
    "MaskHarmonize": "Mask Harmonize",
    "FreeformNeighborToneMatch": "Freeform Neighbor Tone Match",
    "NeighborToneMatch": "Neighbor Tone Match",
    "SeamLatentAnchor": "Seam Latent Anchor",
    "SeamHarmonizerV3": "Seam Harmonizer v3",
}

if Flux2KleinSpatialDenoiseKSamplerNode is not None:
    NODE_CLASS_MAPPINGS["Flux2KleinSpatialDenoiseKSampler"] = Flux2KleinSpatialDenoiseKSamplerNode
    NODE_DISPLAY_NAME_MAPPINGS["Flux2KleinSpatialDenoiseKSampler"] = "Flux2 Klein Spatial Denoise KSampler"

if SeamGuidedKSamplerNode is not None:
    NODE_CLASS_MAPPINGS["SeamGuidedKSampler"] = SeamGuidedKSamplerNode
    NODE_DISPLAY_NAME_MAPPINGS["SeamGuidedKSampler"] = "Seam Guided KSampler"

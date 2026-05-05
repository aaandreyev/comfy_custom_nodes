from __future__ import annotations

from .nodes.mask_harmonize import MaskHarmonize
from .nodes.poisson_inpaint_prefill import PoissonInpaintPrefill
from .nodes.seam_guided_ksampler_node import SeamGuidedKSamplerNode
from .nodes.seam_latent_anchor_node import SeamLatentAnchorNode
from .nodes.seam_harmonizer_node import SeamHarmonizerV3Node


NODE_CLASS_MAPPINGS = {
    "PoissonInpaintPrefill": PoissonInpaintPrefill,
    "MaskHarmonize": MaskHarmonize,
    "SeamGuidedKSampler": SeamGuidedKSamplerNode,
    "SeamLatentAnchor": SeamLatentAnchorNode,
    "SeamHarmonizerV3": SeamHarmonizerV3Node,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PoissonInpaintPrefill": "Poisson Inpaint Prefill",
    "MaskHarmonize": "Mask Harmonize",
    "SeamGuidedKSampler": "Seam Guided KSampler",
    "SeamLatentAnchor": "Seam Latent Anchor",
    "SeamHarmonizerV3": "Seam Harmonizer v3",
}

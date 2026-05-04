from __future__ import annotations

from .nodes.mask_harmonize import MaskHarmonize
from .nodes.poisson_inpaint_prefill import PoissonInpaintPrefill
from .nodes.seam_harmonizer_node import SeamHarmonizerV3Node


NODE_CLASS_MAPPINGS = {
    "PoissonInpaintPrefill": PoissonInpaintPrefill,
    "MaskHarmonize": MaskHarmonize,
    "SeamHarmonizerV3": SeamHarmonizerV3Node,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PoissonInpaintPrefill": "Poisson Inpaint Prefill",
    "MaskHarmonize": "Mask Harmonize",
    "SeamHarmonizerV3": "Seam Harmonizer v3",
}

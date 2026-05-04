from __future__ import annotations

from typing import Iterable

import torch

from ..data.strip_geometry import StripSpec, extract_side_strip


def extract_active_strips(
    image: torch.Tensor,
    bbox: tuple[int, int, int, int],
    sides: Iterable[str],
    inner_width: int,
    outer_width: int = 128,
) -> dict[str, dict]:
    outputs: dict[str, dict] = {}
    spec = StripSpec(outer_width=outer_width, inner_width=inner_width)
    for side in sides:
        strip, meta = extract_side_strip(image, bbox, side, spec)
        outputs[side] = {"strip": strip, "meta": meta}
    return outputs

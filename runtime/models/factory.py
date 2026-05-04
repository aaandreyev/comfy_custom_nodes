from __future__ import annotations

from typing import Any

import torch

from .harmonizer import SeamHarmonizerV3


def build_model_from_config(cfg: dict[str, Any]) -> torch.nn.Module:
    model_cfg = cfg.get("model", cfg.get("architecture", {}))
    arch = model_cfg.get("architecture") or model_cfg.get("name")
    if arch != "seam_harmonizer_v3":
        raise RuntimeError(f"Unsupported architecture: {arch}")
    return SeamHarmonizerV3(
        in_channels=int(model_cfg.get("in_channels", 9)),
        channels=tuple(model_cfg.get("channels", [32, 64, 128, 192])),
        blocks=tuple(model_cfg.get("blocks", [2, 2, 4, 6])),
        outer_width=int((cfg.get("strip") or {}).get("outer_width", 128)),
        boundary_band_px=int((cfg.get("strip") or {}).get("boundary_band_px", 24)),
        correction_limits=model_cfg.get("correction_limits"),
    )

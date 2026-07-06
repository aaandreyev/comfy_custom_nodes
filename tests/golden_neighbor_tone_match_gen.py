"""Generate golden outputs for neighbor tone match regression tests.

Goldens pin the currently accepted implementation; tests compare future
changes against the saved tensors. Regenerate only when an output change is
intentional and visually verified.

History: the 2026-07 speed refactor (EDT LUT fill + bbox crop + GPU port) was
verified against the legacy implementation from git — bit-exact whenever the
LUT has no empty bins (bins<=3 on the test pattern), and within 6e-3 (~1.5/255)
otherwise, entirely attributable to the intended empty-bin fill change
(weighted diffusion -> exact nearest valid bin).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT.parent))

from comfy_custom_nodes_repo.runtime.infer.neighbor_tone_match import (
    apply_freeform_neighbor_tone_match,
    apply_neighbor_tone_match,
)

GOLDEN_PATH = Path(__file__).parent / "golden" / "neighbor_tone_match.pt"


def make_inputs(H: int = 256, W: int = 256) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    yy, xx = torch.meshgrid(torch.arange(H).float(), torch.arange(W).float(), indexing="ij")
    base = 0.5 + 0.35 * torch.sin(2 * torch.pi * xx / 32) * torch.cos(2 * torch.pi * yy / 32)
    ref = torch.stack([base, base.roll(5, dims=1) * 0.8 + 0.1, base.roll(11, dims=0) * 0.6 + 0.2]).unsqueeze(0)
    img = (ref * 0.88 + 0.04).clamp(0, 1)
    rect_mask = torch.zeros(1, 1, H, W)
    rect_mask[:, :, H // 4 : 3 * H // 4, W // 4 : 3 * W // 4] = 1.0
    blob = ((yy - H / 2) ** 2 / (H / 3.2) ** 2 + (xx - W / 2) ** 2 / (W / 4) ** 2) < 1.0
    blob_mask = blob.float().view(1, 1, H, W)
    return ref, img, rect_mask, blob_mask


def build_goldens() -> dict[str, torch.Tensor]:
    ref, img, rect_mask, blob_mask = make_inputs()
    goldens: dict[str, torch.Tensor] = {}
    for lut_mode in ("3d", "2d_luma_curve"):
        for correction_mode in ("hybrid", "additive", "multiplicative"):
            for bins in (3, 8):  # bins=3: dense LUT, no empty-bin fill involved
                key = f"rect_{lut_mode}_{correction_mode}_bins{bins}"
                out, _ = apply_neighbor_tone_match(
                    ref, img, img, rect_mask,
                    inner_width=48, outer_band_px=64, inner_flat_top_px=8,
                    process_left=True, process_right=True, process_top=True, process_bottom=True,
                    luma_strength=1.0, chroma_strength=1.0, u_strength=1.0, v_strength=1.0,
                    bins=bins, correction_mode=correction_mode, lut_mode=lut_mode,
                    color_space="linear", yuv_matrix="bt601",
                    delta_smoothing_sigma=0.0, corner_px=0.0,
                )
                goldens[key] = out
    for lut_mode in ("3d", "2d_luma_curve"):
        key = f"freeform_{lut_mode}_hybrid"
        out, _ = apply_freeform_neighbor_tone_match(
            ref, img, blob_mask,
            inner_width=48, outer_band_px=32, inner_flat_top_px=0,
            luma_strength=1.0, chroma_strength=1.0, u_strength=1.0, v_strength=1.0,
            bins=8, correction_mode="hybrid", lut_mode=lut_mode,
            color_space="linear", yuv_matrix="bt601",
            delta_smoothing_sigma=0.5,
        )
        goldens[key] = out
    out, _ = apply_neighbor_tone_match(
        ref, img, img, rect_mask,
        inner_width=48, outer_band_px=64, inner_flat_top_px=8,
        process_left=True, process_right=True, process_top=True, process_bottom=True,
        luma_strength=1.0, chroma_strength=1.0, u_strength=1.0, v_strength=1.0,
        bins=8, correction_mode="hybrid", lut_mode="3d",
        color_space="srgb", yuv_matrix="bt709",
        delta_smoothing_sigma=2.0, corner_px=None,
    )
    goldens["rect_srgb_bt709_smoothed"] = out
    return goldens


if __name__ == "__main__":
    goldens = build_goldens()
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(goldens, GOLDEN_PATH)
    print(f"saved {len(goldens)} goldens to {GOLDEN_PATH}")
    for k, v in goldens.items():
        print(f"  {k}: {tuple(v.shape)} mean={v.mean():.6f}")

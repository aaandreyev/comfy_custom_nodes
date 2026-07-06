"""Batching contract for image-processing nodes.

Every IMAGE node must accept batch > 1, including the ComfyUI convention of a
batch-1 mask broadcast over a larger image batch, and must process each frame
with its own data (frame 0's correction must never leak into frame 1).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT.parent))

from comfy_custom_nodes_repo.nodes.draw_mask_overlay_advanced_node import DrawMaskOverlayAdvancedNode
from comfy_custom_nodes_repo.nodes.freeform_neighbor_tone_match_node import FreeformNeighborToneMatchNode
from comfy_custom_nodes_repo.nodes.mask_harmonize import MaskHarmonize
from comfy_custom_nodes_repo.nodes.masked_color_transfer_node import MaskedColorTransferNode
from comfy_custom_nodes_repo.nodes.neighbor_tone_match_node import NeighborToneMatchNode
from comfy_custom_nodes_repo.nodes.poisson_inpaint_prefill import PoissonInpaintPrefill
from comfy_custom_nodes_repo.nodes.color_transfer_ref_from_mask_band_node import ColorTransferRefFromMaskBandNode
from comfy_custom_nodes_repo.runtime.infer.correct_full_frame import apply_corrector_to_full_frame
from comfy_custom_nodes_repo.runtime.infer.seam_latent_anchor import (
    apply_seam_anchor_correction,
    prepare_seam_anchor_state,
)
from comfy_custom_nodes_repo.runtime.infer.zero_drift_inpaint_crop import (
    run_zero_drift_crop,
    stitch_zero_drift_result,
)

H = W = 96


def _images(batch: int = 2) -> torch.Tensor:
    gen = torch.Generator().manual_seed(7)
    base = torch.rand(batch, H, W, 3, generator=gen)
    return (base * 0.6 + 0.2).clamp(0, 1)


def _rect_mask(batch: int = 1) -> torch.Tensor:
    mask = torch.zeros(batch, H, W)
    mask[:, H // 4 : 3 * H // 4, W // 4 : 3 * W // 4] = 1.0
    return mask


def _two_masks() -> torch.Tensor:
    masks = torch.zeros(2, H, W)
    masks[0, 10:40, 10:50] = 1.0
    masks[1, 50:90, 30:80] = 1.0
    return masks


def test_poisson_prefill_batch_with_single_mask() -> None:
    images = _images(2)
    (out,) = PoissonInpaintPrefill().process(images, _rect_mask(1))
    assert out.shape == images.shape
    (single,) = PoissonInpaintPrefill().process(images[1:2], _rect_mask(1))
    assert torch.allclose(out[1:2], single, atol=1e-6)


def test_mask_harmonize_batch_with_single_mask() -> None:
    images = _images(2)
    (out,) = MaskHarmonize().harmonize(
        images, _rect_mask(1),
        mode="inside", strip_width=4, blur_sigma=5.0, falloff=16,
        correction_strength=1.0, luminance_strength=1.0, chroma_strength=1.0,
        mask_threshold=0.5, corner_spread=0,
    )
    assert out.shape == images.shape
    (single,) = MaskHarmonize().harmonize(
        images[1:2], _rect_mask(1),
        mode="inside", strip_width=4, blur_sigma=5.0, falloff=16,
        correction_strength=1.0, luminance_strength=1.0, chroma_strength=1.0,
        mask_threshold=0.5, corner_spread=0,
    )
    assert torch.allclose(out[1:2], single, atol=1e-6)


def test_draw_mask_overlay_batch() -> None:
    images = _images(2)
    (out,) = DrawMaskOverlayAdvancedNode().apply(images, _rect_mask(1), "solid_color", "255,0,0", 0.5)
    assert out.shape == images.shape
    assert not torch.equal(out[0], out[1])


def test_masked_color_transfer_batch() -> None:
    images = _images(2)
    ref = _images(2).flip(1)
    (out,) = MaskedColorTransferNode().transfer(
        images, image_ref=ref, mask=_rect_mask(1),
        method="histogram", source_stats="per_frame",
        target_index=0, strength=1.0, mask_threshold=0.5,
    )
    assert out.shape == images.shape
    outside = _rect_mask(1)[0] <= 0.5
    for b in range(2):
        assert torch.equal(out[b][outside], images[b][outside])


def test_color_transfer_ref_from_mask_band_batch() -> None:
    images = _images(2)
    reference, debug_strip = ColorTransferRefFromMaskBandNode().build(
        images, _rect_mask(1), outer_band_px=12, min_output_size=16,
        max_output_size=64, mask_threshold=0.5,
    )
    assert reference.shape[0] == 2
    assert debug_strip.shape[0] == 2
    assert not torch.equal(reference[0], reference[1])


def test_zero_drift_crop_stitch_batch_different_masks_roundtrip() -> None:
    images = _images(2)
    masks = _two_masks()
    # align=False: no crop resampling, so an unchanged crop must stitch back exactly
    stitcher, cropped, cropped_mask = run_zero_drift_crop(
        image=images,
        downscale_algorithm="bilinear",
        upscale_algorithm="bicubic",
        mask_expand_pixels=0,
        mask_blend_pixels=4,
        context_from_mask_extend_factor=1.2,
        mask=masks,
        optional_context_mask=None,
        align_crop_spatial_multiple_of_8=False,
    )
    assert cropped.shape[0] == 2
    assert cropped_mask.shape[0] == 2
    assert not torch.equal(cropped_mask[0], cropped_mask[1])
    restored = stitch_zero_drift_result(stitcher, cropped.clone())
    assert restored.shape == images.shape
    assert torch.allclose(restored, images, atol=1e-6)

    # align=True: crop may be resampled to a multiple of 8, which bleeds a few
    # boundary pixels even at batch 1. Pixels beyond the mask + blend margin
    # must still survive the roundtrip untouched for every batch element.
    stitcher8, cropped8, _ = run_zero_drift_crop(
        image=images,
        downscale_algorithm="bilinear",
        upscale_algorithm="bicubic",
        mask_expand_pixels=0,
        mask_blend_pixels=4,
        context_from_mask_extend_factor=1.2,
        mask=masks,
        optional_context_mask=None,
        align_crop_spatial_multiple_of_8=True,
    )
    assert cropped8.shape[0] == 2
    assert cropped8.shape[1] % 8 == 0 and cropped8.shape[2] % 8 == 0
    restored8 = stitch_zero_drift_result(stitcher8, cropped8.clone())
    dilated = torch.nn.functional.max_pool2d(masks.unsqueeze(1), kernel_size=13, stride=1, padding=6)[:, 0]
    for b in range(2):
        far_outside = dilated[b] <= 0.5
        assert torch.equal(restored8[b][far_outside], images[b][far_outside])


def test_zero_drift_crop_batch_same_mask_matches_single() -> None:
    images = _images(2)
    mask = _rect_mask(1)
    kwargs = dict(
        downscale_algorithm="bilinear",
        upscale_algorithm="bicubic",
        mask_expand_pixels=0,
        mask_blend_pixels=4,
        context_from_mask_extend_factor=1.2,
        optional_context_mask=None,
        align_crop_spatial_multiple_of_8=True,
    )
    _, cropped_b, _ = run_zero_drift_crop(image=images, mask=mask, **kwargs)
    _, cropped_s, _ = run_zero_drift_crop(image=images[1:2], mask=mask, **kwargs)
    assert torch.allclose(cropped_b[1:2], cropped_s, atol=1e-6)


class _StubHarmonizer(torch.nn.Module):
    """Returns a per-batch-element constant delta so frame leakage is detectable."""

    outer_width = 16
    boundary_band_px = 4

    def __init__(self) -> None:
        super().__init__()
        self.register_parameter("dummy", torch.nn.Parameter(torch.zeros(1)))

    def forward(self, model_in: torch.Tensor) -> dict[str, torch.Tensor]:
        inner = model_in[:, :3, :, self.outer_width :]
        shifts = torch.arange(inner.shape[0], device=inner.device, dtype=inner.dtype)
        shifts = (shifts + 1.0).view(-1, 1, 1, 1) * 0.01
        low = torch.zeros(inner.shape[0], 1, 4, 4, device=inner.device, dtype=inner.dtype)
        return {
            "corrected_inner": inner + shifts,
            "confidence": torch.ones_like(inner[:, :1]),
            "gain_lowres": low, "gamma_lowres": low, "bias_lowres": low,
            "detail_lowres": low, "gate_lowres": low,
        }


def test_seam_harmonizer_corrects_each_frame_from_its_own_strips() -> None:
    images = _images(2).permute(0, 3, 1, 2)
    mask = _rect_mask(1).unsqueeze(1)
    bbox = (W // 4, H // 4, 3 * W // 4, 3 * H // 4)
    corrected, _ = apply_corrector_to_full_frame(
        _StubHarmonizer(), images, mask, bbox, ["left", "right"], inner_width=8,
    )
    assert corrected.shape == images.shape
    delta0 = (corrected[0] - images[0]).abs().sum()
    delta1 = (corrected[1] - images[1]).abs().sum()
    assert delta0 > 0 and delta1 > 0
    # stub shifts grow with strip batch index: frame 1 must differ from frame 0
    assert not torch.allclose(corrected[0] - images[0], corrected[1] - images[1])


def test_seam_latent_anchor_state_batch_with_single_mask() -> None:
    latents = torch.rand(2, 4, 32, 32)
    state = prepare_seam_anchor_state(
        latents, _rect_mask(1)[:, ::3, ::3],
        anchor_width_px=2, anchor_falloff_px=4,
    )
    denoised = torch.rand(2, 4, 32, 32)
    corrected = apply_seam_anchor_correction(denoised, state, 0.5)
    assert corrected.shape == denoised.shape


@pytest.mark.parametrize("node_cls, extra", [
    (NeighborToneMatchNode, {
        "inner_flat_top_px": 4,
        "process_left": True, "process_right": True,
        "process_top": True, "process_bottom": True,
        "corner_px": 0,
    }),
    (FreeformNeighborToneMatchNode, {"inner_flat_top_px": 0}),
])
def test_tone_match_nodes_batch_with_single_mask(node_cls, extra) -> None:
    images = _images(2)
    ref = (images * 0.9 + 0.03).clamp(0, 1)
    (out,) = node_cls().run(
        reference_image=ref,
        image=images,
        mask=_rect_mask(1),
        inner_width=16,
        outer_band_px=16,
        luma_strength=1.0, chroma_strength=1.0, u_strength=1.0, v_strength=1.0,
        bins=8, correction_mode="hybrid", lut_mode="3d",
        color_space="linear", yuv_matrix="bt601",
        delta_smoothing_sigma=0.0,
        debug_previews=False,
        **extra,
    )
    assert out.shape == images.shape
    outside = _rect_mask(1)[0] <= 0.5
    for b in range(2):
        assert torch.equal(out[b][outside], images[b][outside])
        assert not torch.equal(out[b][~outside], images[b][~outside])

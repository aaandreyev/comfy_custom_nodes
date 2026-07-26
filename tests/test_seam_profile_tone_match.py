"""Tests for the seam-profile tone correction.

The scenario every test builds is the one the LUT node structurally cannot fix:
a tone error that exists only INSIDE the stitch mask (diffusion drift), so the
donor ring outside carries no signal about it. The seam-profile correction must
cancel the cross-seam step it can measure directly at the boundary.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.ndimage import distance_transform_edt, gaussian_filter, grey_dilation

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT.parent))

from comfy_custom_nodes_repo.nodes.seam_profile_tone_match_node import SeamProfileToneMatchNode
from comfy_custom_nodes_repo.runtime.infer.seam_profile_tone_match import apply_seam_profile_tone_match

SIZE = 256


def _base_image() -> np.ndarray:
    ramp = np.linspace(0.35, 0.65, SIZE, dtype=np.float32)
    gray = np.tile(ramp, (SIZE, 1))
    return np.stack([gray, gray, gray], axis=0)


def _circle_mask(radius: int = 60) -> np.ndarray:
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    return (np.hypot(yy - SIZE / 2, xx - SIZE / 2) <= radius).astype(np.float32)


def _biased_image(base: np.ndarray, mask: np.ndarray, bias) -> np.ndarray:
    feather = gaussian_filter(mask, 2.0)
    return np.clip(base + np.asarray(bias, dtype=np.float32) * feather[None], 0.0, 1.0)


def _to_bchw(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(img).unsqueeze(0).float()


def _cross_seam_step(final: np.ndarray, reference: np.ndarray, mask: np.ndarray, region=None) -> float:
    """Signed low-passed luma step between the 1..4px inner band and 1..4px outer band."""
    mask_bool = mask > 0.5
    dist_in = distance_transform_edt(mask_bool)
    dist_out = distance_transform_edt(~mask_bool)
    lp_f = gaussian_filter(final[0], 2.0)
    lp_r = gaussian_filter(reference[0], 2.0)
    band_in = mask_bool & (dist_in >= 1) & (dist_in <= 4)
    band_out = (~mask_bool) & (dist_out >= 1) & (dist_out <= 4)
    if region is not None:
        band_in = band_in & region
        band_out = band_out & region
    return float(lp_f[band_in].mean() - lp_r[band_out].mean())


def _run(reference, image, mask, **overrides):
    params = dict(inner_width=48, color_space="linear", yuv_matrix="bt601")
    params.update(overrides)
    corrected, debug = apply_seam_profile_tone_match(
        _to_bchw(reference), _to_bchw(image), torch.from_numpy(mask).unsqueeze(0), **params
    )
    return corrected[0].numpy(), debug


def test_identity_is_noop() -> None:
    base = _base_image()
    mask = _circle_mask()
    corrected, debug = _run(base, base.copy(), mask)
    assert debug["reason"] == "applied"
    assert np.abs(corrected - base).max() < 2e-3


def test_outside_mask_untouched() -> None:
    base = _base_image()
    mask = _circle_mask()
    image = _biased_image(base, mask, -0.06)
    corrected, _ = _run(base, image, mask)
    outside = mask <= 0.5
    assert np.array_equal(corrected[:, outside], image[:, outside])


def test_interior_only_bias_step_removed() -> None:
    base = _base_image()
    mask = _circle_mask()
    image = _biased_image(base, mask, -0.06)
    step_before = _cross_seam_step(image, base, mask)
    corrected, debug = _run(base, image, mask)
    step_after = _cross_seam_step(corrected, base, mask)
    assert debug["reason"] == "applied"
    assert abs(step_before) > 0.04
    assert abs(step_after) < 0.3 * abs(step_before)


def test_bias_varying_along_seam_removed_locally() -> None:
    base = _base_image()
    mask = _circle_mask(radius=60)
    xx = np.tile(np.arange(SIZE, dtype=np.float32), (SIZE, 1))
    ramp = -0.08 + 0.16 * ((xx - (SIZE / 2 - 60)) / 120.0).clip(0.0, 1.0)
    image = _biased_image(base, mask, ramp)
    left = xx < SIZE / 2 - 30
    right = xx > SIZE / 2 + 30
    corrected, _ = _run(base, image, mask)
    for region in (left, right):
        before = _cross_seam_step(image, base, mask, region)
        after = _cross_seam_step(corrected, base, mask, region)
        assert abs(before) > 0.03
        assert abs(after) < 0.4 * abs(before)


def test_correction_fades_out_deep_inside() -> None:
    base = _base_image()
    mask = _circle_mask()
    image = _biased_image(base, mask, -0.06)
    corrected, _ = _run(base, image, mask, inner_width=24)
    dist_in = distance_transform_edt(mask > 0.5)
    deep = dist_in > 30
    assert deep.any()
    assert np.abs(corrected[:, deep] - image[:, deep]).max() < 1e-6


def test_max_correction_clamps_applied_delta() -> None:
    base = _base_image()
    mask = _circle_mask()
    image = _biased_image(base, mask, -0.2)
    corrected, _ = _run(base, image, mask, max_correction=0.05)
    applied = np.abs(corrected[0] - image[0]).max()
    assert applied < 0.05 + 1e-3


def test_seam_inset_measures_the_eroded_boundary() -> None:
    base = _base_image()
    mask = _circle_mask(radius=60)
    grown = grey_dilation(mask, size=(31, 31))
    image = _biased_image(base, mask, -0.06)
    corrected, _ = _run(base, image, grown, seam_inset_px=15)
    before = _cross_seam_step(image, base, mask)
    after = _cross_seam_step(corrected, base, mask)
    assert abs(after) < 0.35 * abs(before)


def test_batch_elements_are_independent() -> None:
    base = _base_image()
    mask = _circle_mask()
    img_dark = _biased_image(base, mask, -0.06)
    img_bright = _biased_image(base, mask, 0.06)
    reference = torch.from_numpy(np.stack([base, base])).float()
    image = torch.from_numpy(np.stack([img_dark, img_bright])).float()
    masks = torch.from_numpy(np.stack([mask, mask]))
    corrected, debug = apply_seam_profile_tone_match(
        reference, image, masks, inner_width=48, color_space="linear", yuv_matrix="bt601"
    )
    assert debug["reason"] == "applied"
    assert torch.isfinite(corrected).all()
    seam_band = torch.from_numpy((distance_transform_edt(mask > 0.5) <= 4) & (mask > 0.5))
    assert (corrected[0, 0][seam_band] - image[0, 0][seam_band]).mean() > 0.02
    assert (corrected[1, 0][seam_band] - image[1, 0][seam_band]).mean() < -0.02


def test_empty_mask_is_noop_with_reason() -> None:
    base = _base_image()
    corrected, debug = _run(base, base.copy(), np.zeros((SIZE, SIZE), dtype=np.float32))
    assert debug["reason"] == "empty_mask"
    assert np.array_equal(corrected, base)


def test_bfloat16_inputs_supported() -> None:
    base = _base_image()
    mask = _circle_mask()
    image = _biased_image(base, mask, -0.06)
    corrected, debug = apply_seam_profile_tone_match(
        _to_bchw(base).to(torch.bfloat16),
        _to_bchw(image).to(torch.bfloat16),
        torch.from_numpy(mask).unsqueeze(0),
        inner_width=48,
        color_space="linear",
        yuv_matrix="bt601",
    )
    assert corrected.dtype == torch.bfloat16
    assert debug["reason"] == "applied"


def test_node_wrapper_runs_and_matches_core() -> None:
    base = _base_image()
    mask = _circle_mask()
    image = _biased_image(base, mask, -0.06)
    node = SeamProfileToneMatchNode()
    (out,) = node.run(
        reference_image=torch.from_numpy(base).permute(1, 2, 0).unsqueeze(0).float(),
        image=torch.from_numpy(image).permute(1, 2, 0).unsqueeze(0).float(),
        mask=torch.from_numpy(mask).unsqueeze(0),
        inner_width=48,
        inner_flat_top_px=0,
        seam_inset_px=0,
        lowpass_sigma=3.0,
        arc_smooth_px=12.0,
        max_correction=0.5,
        luma_strength=1.0,
        chroma_strength=1.0,
        u_strength=1.0,
        v_strength=1.0,
        color_space="linear",
        yuv_matrix="bt601",
        debug_previews=False,
    )
    core, _ = _run(base, image, mask)
    assert out.shape == (1, SIZE, SIZE, 3)
    assert np.abs(out[0].numpy().transpose(2, 0, 1) - core).max() < 1e-6


def test_node_wrapper_rejects_empty_mask() -> None:
    base = _base_image()
    node = SeamProfileToneMatchNode()
    with pytest.raises(ValueError, match="no active pixels"):
        node.run(
            reference_image=torch.from_numpy(base).permute(1, 2, 0).unsqueeze(0).float(),
            image=torch.from_numpy(base).permute(1, 2, 0).unsqueeze(0).float(),
            mask=torch.zeros(1, SIZE, SIZE),
            inner_width=48,
            inner_flat_top_px=0,
            seam_inset_px=0,
            lowpass_sigma=3.0,
            arc_smooth_px=12.0,
            max_correction=0.5,
            luma_strength=1.0,
            chroma_strength=1.0,
            u_strength=1.0,
            v_strength=1.0,
            color_space="linear",
            yuv_matrix="bt601",
            debug_previews=False,
        )

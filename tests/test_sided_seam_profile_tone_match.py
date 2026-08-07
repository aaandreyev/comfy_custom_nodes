"""Tests for the sided seam-profile correction (NeighborToneMatch placement).

Outpaint scenario: the mask is the newly generated region touching canvas
edges; only sides with real neighbor content may drive the correction, and the
falloff must be measured from the valid seam, never from canvas borders.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.ndimage import distance_transform_edt, gaussian_filter

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT.parent))

from comfy_custom_nodes_repo.nodes.sided_seam_profile_tone_match_node import SidedSeamProfileToneMatchNode
from comfy_custom_nodes_repo.runtime.infer.seam_profile_tone_match import (
    apply_sided_seam_profile_tone_match,
)

SIZE = 256


def _base_image() -> np.ndarray:
    ramp = np.linspace(0.35, 0.65, SIZE, dtype=np.float32)
    gray = np.tile(ramp, (SIZE, 1))
    return np.stack([gray, gray, gray], axis=0)


def _right_half_mask() -> np.ndarray:
    mask = np.zeros((SIZE, SIZE), dtype=np.float32)
    mask[:, SIZE // 2 :] = 1.0
    return mask


def _center_strip_mask() -> np.ndarray:
    mask = np.zeros((SIZE, SIZE), dtype=np.float32)
    mask[:, 96:160] = 1.0
    return mask


def _biased_image(base: np.ndarray, mask: np.ndarray, bias) -> np.ndarray:
    feather = gaussian_filter(mask, 2.0)
    return np.clip(base + np.asarray(bias, dtype=np.float32) * feather[None], 0.0, 1.0)


def _to_bchw(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(img).unsqueeze(0).float()


def _left_seam_step(final: np.ndarray, reference: np.ndarray, mask: np.ndarray, x_seam: int) -> float:
    lp_f = gaussian_filter(final[0], 2.0)
    lp_r = gaussian_filter(reference[0], 2.0)
    inner = lp_f[:, x_seam + 1 : x_seam + 5].mean()
    outer = lp_r[:, x_seam - 5 : x_seam - 1].mean()
    return float(inner - outer)


def _run(reference, image, mask, topology=None, **overrides):
    params = dict(inner_width=32, color_space="linear", yuv_matrix="bt601")
    params.update(overrides)
    corrected, debug = apply_sided_seam_profile_tone_match(
        _to_bchw(reference), _to_bchw(image), torch.from_numpy(mask).unsqueeze(0),
        topology_mask=topology, **params,
    )
    return corrected[0].numpy(), debug


def test_identity_is_noop() -> None:
    base = _base_image()
    corrected, debug = _run(base, base.copy(), _right_half_mask())
    assert debug["reason"] == "applied"
    assert np.abs(corrected - base).max() < 2e-3


def test_outpaint_bias_removed_at_left_seam_only() -> None:
    base = _base_image()
    mask = _right_half_mask()
    image = _biased_image(base, mask, -0.06)
    corrected, debug = _run(base, image, mask)
    assert debug["reason"] == "applied"
    assert debug["per_sample"][0]["sides"] == ["left"]
    before = _left_seam_step(image, base, mask, SIZE // 2)
    after = _left_seam_step(corrected, base, mask, SIZE // 2)
    assert abs(before) > 0.04
    assert abs(after) < 0.3 * abs(before)
    far = slice(SIZE // 2 + 32 + 12, SIZE)
    assert np.abs(corrected[:, :, far] - image[:, :, far]).max() < 1e-6


def test_outside_mask_untouched() -> None:
    base = _base_image()
    mask = _right_half_mask()
    image = _biased_image(base, mask, -0.06)
    corrected, _ = _run(base, image, mask)
    outside = mask <= 0.5
    assert np.array_equal(corrected[:, outside], image[:, outside])


def test_disabled_side_makes_node_noop() -> None:
    base = _base_image()
    mask = _right_half_mask()
    image = _biased_image(base, mask, -0.06)
    corrected, debug = _run(base, image, mask, process_left=False)
    assert debug["reason"] == "no_processable_sides"
    assert np.array_equal(corrected, image)


def test_topology_mask_filters_sides() -> None:
    base = _base_image()
    mask = _center_strip_mask()
    image = _biased_image(base, mask, -0.06)

    topo_west = torch.zeros(1, SIZE, SIZE)
    topo_west[:, :, :96] = 1.0
    corrected, debug = _run(base, image, mask, topology=topo_west)
    assert debug["per_sample"][0]["sides"] == ["left"]
    left_after = _left_seam_step(corrected, base, mask, 96)
    assert abs(left_after) < 0.02

    right_inner = np.abs(corrected[0][:, 156:159] - image[0][:, 156:159]).mean()
    corrected_both, debug_both = _run(base, image, mask)
    assert set(debug_both["per_sample"][0]["sides"]) == {"left", "right"}
    right_inner_both = np.abs(corrected_both[0][:, 156:159] - image[0][:, 156:159]).mean()
    assert right_inner_both > right_inner


def test_both_seams_corrected_for_center_strip() -> None:
    base = _base_image()
    mask = _center_strip_mask()
    image = _biased_image(base, mask, -0.06)
    corrected, debug = _run(base, image, mask)
    assert set(debug["per_sample"][0]["sides"]) == {"left", "right"}
    lp_c = gaussian_filter(corrected[0], 2.0)
    lp_i = gaussian_filter(image[0], 2.0)
    for band in (slice(97, 100), slice(156, 159)):
        assert lp_c[:, band].mean() > lp_i[:, band].mean() + 0.02


def test_bfloat16_supported() -> None:
    base = _base_image()
    mask = _right_half_mask()
    image = _biased_image(base, mask, -0.06)
    corrected, debug = apply_sided_seam_profile_tone_match(
        _to_bchw(base).to(torch.bfloat16),
        _to_bchw(image).to(torch.bfloat16),
        torch.from_numpy(mask).unsqueeze(0),
        inner_width=32,
        color_space="linear",
        yuv_matrix="bt601",
    )
    assert corrected.dtype == torch.bfloat16
    assert debug["reason"] == "applied"


def _node_kwargs():
    return dict(
        inner_width=32, inner_flat_top_px=0, seam_inset_px=0,
        process_left=True, process_right=True, process_top=True, process_bottom=True,
        lowpass_sigma=3.0, arc_smooth_px=12.0, max_correction=0.5,
        luma_strength=1.0, chroma_strength=1.0, u_strength=1.0, v_strength=1.0,
        color_space="linear", yuv_matrix="bt601", debug_previews=False,
    )


def test_node_wrapper_matches_core() -> None:
    base = _base_image()
    mask = _right_half_mask()
    image = _biased_image(base, mask, -0.06)
    node = SidedSeamProfileToneMatchNode()
    (out,) = node.run(
        reference_image=torch.from_numpy(base).permute(1, 2, 0).unsqueeze(0).float(),
        image=torch.from_numpy(image).permute(1, 2, 0).unsqueeze(0).float(),
        mask=torch.from_numpy(mask).unsqueeze(0),
        **_node_kwargs(),
    )
    core, _ = _run(base, image, mask)
    assert np.abs(out[0].numpy().transpose(2, 0, 1) - core).max() < 1e-6


def test_node_wrapper_rejects_empty_mask() -> None:
    base = _base_image()
    node = SidedSeamProfileToneMatchNode()
    with pytest.raises(ValueError, match="no active pixels"):
        node.run(
            reference_image=torch.from_numpy(base).permute(1, 2, 0).unsqueeze(0).float(),
            image=torch.from_numpy(base).permute(1, 2, 0).unsqueeze(0).float(),
            mask=torch.zeros(1, SIZE, SIZE),
            **_node_kwargs(),
        )


def test_batched_image_with_single_reference() -> None:
    base = _base_image()
    mask = _right_half_mask()
    imgs = np.stack([_biased_image(base, mask, -0.06), _biased_image(base, mask, 0.05)])
    corrected, debug = apply_sided_seam_profile_tone_match(
        _to_bchw(base),
        torch.from_numpy(imgs).float(),
        torch.from_numpy(mask).unsqueeze(0),
        inner_width=32,
        color_space="linear",
        yuv_matrix="bt601",
    )
    assert debug["reason"] == "applied"
    assert corrected.shape[0] == 2
    band = torch.zeros(SIZE, SIZE, dtype=torch.bool)
    band[:, SIZE // 2 : SIZE // 2 + 4] = True
    imgs_t = torch.from_numpy(imgs).float()
    assert (corrected[0, 0][band] - imgs_t[0, 0][band]).mean() > 0.005
    assert (corrected[1, 0][band] - imgs_t[1, 0][band]).mean() < -0.005


def test_topology_mask_with_no_real_content_is_noop() -> None:
    base = _base_image()
    mask = _center_strip_mask()
    image = _biased_image(base, mask, -0.06)
    # Topology supplied but empty: authoritative "no real neighbors" -> no-op.
    # Feeding the generation mask itself as topology reduces to the same case.
    for topo in (torch.zeros(1, SIZE, SIZE), torch.from_numpy(mask).unsqueeze(0)):
        corrected, debug = _run(base, image, mask, topology=topo)
        assert debug["reason"] == "no_processable_sides"
        assert np.array_equal(corrected, image)

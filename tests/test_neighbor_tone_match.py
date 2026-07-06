"""Regression and correctness tests for the optimized neighbor tone match path.

Golden tensors in ``golden/neighbor_tone_match.pt`` were produced by the
pre-optimization implementation (iterative LUT fill, full-frame math) via
``golden_neighbor_tone_match_gen.py``. Inputs are periodic so every queried LUT
bin is populated and outputs must not depend on the empty-bin fill strategy.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT.parent))

from comfy_custom_nodes_repo.nodes.freeform_neighbor_tone_match_node import FreeformNeighborToneMatchNode
from comfy_custom_nodes_repo.nodes.neighbor_tone_match_node import NeighborToneMatchNode
from comfy_custom_nodes_repo.runtime.infer.neighbor_tone_match import (
    _nearest_valid_fill,
    apply_freeform_neighbor_tone_match,
    apply_neighbor_tone_match,
)
from comfy_custom_nodes_repo.tests.golden_neighbor_tone_match_gen import GOLDEN_PATH, build_goldens, make_inputs


@pytest.fixture(scope="module")
def goldens() -> dict[str, torch.Tensor]:
    assert GOLDEN_PATH.exists(), "golden file missing; regenerate from the reference implementation"
    return torch.load(GOLDEN_PATH, weights_only=True)


def test_matches_reference_goldens(goldens: dict[str, torch.Tensor]) -> None:
    current = build_goldens()
    assert set(current) == set(goldens)
    for key, expected in goldens.items():
        got = current[key]
        max_diff = (got - expected).abs().max().item()
        assert max_diff < 1e-4, f"{key}: max diff {max_diff}"


@pytest.mark.parametrize("grid_shape", [(9,), (7, 9), (6, 7, 8)])
def test_nearest_valid_fill_is_exact_nearest(grid_shape: tuple[int, ...]) -> None:
    gen = torch.Generator().manual_seed(42)
    valid = torch.rand(1, *grid_shape, generator=gen) < 0.12
    valid[(0,) + tuple(0 for _ in grid_shape)] = True
    values = torch.rand(1, *grid_shape, 3, generator=gen)
    global_value = torch.zeros(1, 3)

    filled = _nearest_valid_fill(values, valid, global_value)

    valid_cells = valid[0].nonzero(as_tuple=False)
    for cell in itertools.product(*(range(s) for s in grid_shape)):
        if valid[0][cell]:
            assert torch.equal(filled[0][cell], values[0][cell])
            continue
        dists = ((valid_cells.float() - torch.tensor(cell, dtype=torch.float32)) ** 2).sum(dim=1)
        min_dist = dists.min()
        candidates = valid_cells[dists == min_dist]
        matches = any(
            torch.allclose(filled[0][cell], values[0][tuple(c.tolist())]) for c in candidates
        )
        assert matches, f"cell {cell} not filled from a nearest valid cell"


def test_nearest_valid_fill_global_fallback() -> None:
    values = torch.rand(1, 4, 4, 3)
    valid = torch.zeros(1, 4, 4, dtype=torch.bool)
    global_value = torch.tensor([[0.5, -0.25, 0.125]])
    filled = _nearest_valid_fill(values, valid, global_value)
    assert torch.allclose(filled, global_value.view(1, 1, 1, 3).expand_as(filled))


def test_rect_untouched_outside_mask() -> None:
    ref, img, rect_mask, _ = make_inputs()
    out, debug = apply_neighbor_tone_match(
        ref, img, img, rect_mask,
        inner_width=48, outer_band_px=64, inner_flat_top_px=8,
        process_left=True, process_right=True, process_top=True, process_bottom=True,
        luma_strength=1.0, chroma_strength=1.0,
        bins=16, correction_mode="hybrid", lut_mode="3d",
        color_space="linear", yuv_matrix="bt601",
        delta_smoothing_sigma=0.0, corner_px=0.0,
    )
    assert debug["reason"] == "applied"
    outside = rect_mask < 0.5
    assert torch.equal(out.masked_select(outside.expand_as(out)), img.masked_select(outside.expand_as(img)))
    inside = rect_mask > 0.5
    assert not torch.equal(out.masked_select(inside.expand_as(out)), img.masked_select(inside.expand_as(img)))


def test_freeform_untouched_outside_mask_and_border_mask() -> None:
    ref, img, _, blob_mask = make_inputs()
    border_mask = blob_mask.clone()
    border_mask[:, :, :, :8] = 1.0  # mask touches the left frame edge
    out, debug = apply_freeform_neighbor_tone_match(
        ref, img, border_mask,
        inner_width=32, outer_band_px=24, inner_flat_top_px=0,
        luma_strength=1.0, chroma_strength=1.0,
        bins=16, correction_mode="hybrid", lut_mode="3d",
        color_space="linear", yuv_matrix="bt601",
        delta_smoothing_sigma=0.5,
    )
    assert debug["reason"] == "applied"
    outside = border_mask < 0.5
    assert torch.equal(out.masked_select(outside.expand_as(out)), img.masked_select(outside.expand_as(img)))


def test_rect_full_frame_mask_passthrough() -> None:
    ref, img, _, _ = make_inputs(128, 128)
    full_mask = torch.ones(1, 1, 128, 128)
    out, debug = apply_neighbor_tone_match(
        ref, img, img, full_mask,
        inner_width=32, outer_band_px=32, inner_flat_top_px=0,
        process_left=True, process_right=True, process_top=True, process_bottom=True,
        luma_strength=1.0, chroma_strength=1.0,
        bins=8, correction_mode="hybrid", lut_mode="3d",
        color_space="linear", yuv_matrix="bt601",
        delta_smoothing_sigma=0.0, corner_px=0.0,
    )
    assert debug["reason"] == "no_processable_sides"
    assert torch.equal(out, img)


@pytest.mark.parametrize("node_cls, extra", [
    (NeighborToneMatchNode, {
        "inner_flat_top_px": 8,
        "process_left": True, "process_right": True,
        "process_top": True, "process_bottom": True,
        "corner_px": 0,
    }),
    (FreeformNeighborToneMatchNode, {"inner_flat_top_px": 0}),
])
def test_node_wrappers_preserve_alpha_and_device(node_cls, extra) -> None:
    ref, img, rect_mask, _ = make_inputs(128, 128)
    ref_bhwc = ref.permute(0, 2, 3, 1)
    img_rgba = torch.cat([img, torch.full((1, 1, 128, 128), 0.7)], dim=1).permute(0, 2, 3, 1)
    ref_rgba = torch.cat([ref, torch.full((1, 1, 128, 128), 0.7)], dim=1).permute(0, 2, 3, 1)
    assert ref_bhwc.shape[-1] == 3
    (out,) = node_cls().run(
        reference_image=ref_rgba,
        image=img_rgba,
        mask=rect_mask[:, 0],
        inner_width=32,
        outer_band_px=32,
        luma_strength=1.0, chroma_strength=1.0, u_strength=1.0, v_strength=1.0,
        bins=8, correction_mode="hybrid", lut_mode="3d",
        color_space="linear", yuv_matrix="bt601",
        delta_smoothing_sigma=0.0,
        debug_previews=False,
        **extra,
    )
    assert out.shape == img_rgba.shape
    assert out.device == img_rgba.device
    assert torch.allclose(out[..., 3], torch.full((1, 128, 128), 0.7))

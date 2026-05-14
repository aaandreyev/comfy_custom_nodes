from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT.parent))

from comfy_custom_nodes_repo.nodes.color_transfer_ref_from_mask_band_node import ColorTransferRefFromMaskBandNode
from comfy_custom_nodes_repo.nodes.draw_mask_overlay_advanced_node import DrawMaskOverlayAdvancedNode
from comfy_custom_nodes_repo.runtime.infer.mask_band_reference import (
    build_mask_band_reference_outputs,
    build_mask_band_reference_image,
    extract_mask_neighborhood_pixels,
    extract_mask_neighborhood_strip_rgba,
)


def test_draw_mask_overlay_advanced_matches_solid_color_fill_with_opacity() -> None:
    node = DrawMaskOverlayAdvancedNode()
    image = torch.zeros((1, 4, 4, 3), dtype=torch.float32)
    mask = torch.zeros((1, 4, 4), dtype=torch.float32)
    mask[:, 1:3, 1:3] = 1.0

    (out,) = node.apply(image, mask, "solid_color", "255, 0, 0", 0.5, "cpu")

    expected = torch.zeros_like(image)
    expected[:, 1:3, 1:3, 0] = 0.5
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-6)


def test_draw_mask_overlay_advanced_grayscale_mode_uses_mask_preview_opacity() -> None:
    node = DrawMaskOverlayAdvancedNode()
    image = torch.full((1, 2, 2, 3), 0.2, dtype=torch.float32)
    mask = torch.tensor([[[0.0, 1.0], [0.5, 0.25]]], dtype=torch.float32)

    (out,) = node.apply(image, mask, "mask_grayscale", "255, 255, 255", 0.5, "cpu")

    expected = torch.empty_like(out)
    for y in range(2):
        for x in range(2):
            value = float(mask[0, y, x].item())
            expected[0, y, x] = 0.2 * 0.5 + value * 0.5
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-6)


def test_extract_mask_neighborhood_pixels_selects_outside_band_only() -> None:
    image = torch.arange(25, dtype=torch.float32).view(1, 5, 5, 1).expand(1, 5, 5, 3) / 24.0
    mask = torch.zeros((1, 5, 5), dtype=torch.float32)
    mask[:, 2, 2] = 1.0

    pixels = extract_mask_neighborhood_pixels(image, mask, outer_band_px=1)

    assert len(pixels) == 1
    assert pixels[0].shape[0] == 4
    expected_positions = {(1, 2), (2, 1), (2, 3), (3, 2)}
    expected_values = {float(image[0, y, x, 0].item()) for y, x in expected_positions}
    actual_values = {float(v.item()) for v in pixels[0][:, 0]}
    assert actual_values == expected_values


def test_build_mask_band_reference_image_packs_pixels_without_empty_zones() -> None:
    image = torch.zeros((1, 8, 8, 3), dtype=torch.float32)
    image[0, :, :, 0] = torch.arange(64, dtype=torch.float32).view(8, 8) / 63.0
    mask = torch.zeros((1, 8, 8), dtype=torch.float32)
    mask[:, 2:6, 2:6] = 1.0

    reference = build_mask_band_reference_image(
        image,
        mask,
        outer_band_px=1,
        min_output_size=4,
        max_output_size=8,
    )

    assert reference.shape == (1, 4, 4, 3)
    assert torch.count_nonzero(reference[0, :, :, 0]) > 0


def test_extract_mask_neighborhood_strip_rgba_keeps_only_outer_band_visible() -> None:
    image = torch.zeros((1, 5, 5, 3), dtype=torch.float32)
    image[0, :, :, 0] = torch.arange(25, dtype=torch.float32).view(5, 5) / 24.0
    mask = torch.zeros((1, 5, 5), dtype=torch.float32)
    mask[:, 2, 2] = 1.0

    strip = extract_mask_neighborhood_strip_rgba(image, mask, outer_band_px=1)

    assert strip.shape == (1, 5, 5, 4)
    expected_positions = {(1, 2), (2, 1), (2, 3), (3, 2)}
    alpha_positions = {
        (y, x)
        for y in range(5)
        for x in range(5)
        if float(strip[0, y, x, 3].item()) > 0.5
    }
    assert alpha_positions == expected_positions
    assert float(strip[0, 0, 0, 3].item()) == 0.0
    assert float(strip[0, 2, 2, 3].item()) == 0.0
    for y, x in expected_positions:
        torch.testing.assert_close(strip[0, y, x, 0], image[0, y, x, 0], atol=1e-6, rtol=1e-6)


def test_build_mask_band_reference_outputs_returns_reference_and_debug_strip() -> None:
    image = torch.zeros((1, 8, 8, 3), dtype=torch.float32)
    image[0, :, :, 1] = 0.7
    mask = torch.zeros((1, 8, 8), dtype=torch.float32)
    mask[:, 2:6, 2:6] = 1.0

    reference, debug_strip = build_mask_band_reference_outputs(
        image,
        mask,
        outer_band_px=1,
        min_output_size=4,
        max_output_size=8,
    )

    assert reference.shape == (1, 4, 4, 3)
    assert debug_strip.shape == (1, 8, 8, 4)
    assert float(debug_strip[0, 0, 0, 3].item()) == 0.0
    assert float(debug_strip[0, 1, 2, 3].item()) == 1.0
    assert float(debug_strip[0, 2, 2, 3].item()) == 0.0


def test_color_transfer_ref_from_mask_band_node_returns_batch_image_and_debug_strip() -> None:
    node = ColorTransferRefFromMaskBandNode()
    image = torch.zeros((2, 16, 16, 3), dtype=torch.float32)
    image[0, :, :, 0] = 0.8
    image[1, :, :, 1] = 0.6
    mask = torch.zeros((2, 16, 16), dtype=torch.float32)
    mask[0, 4:12, 4:12] = 1.0
    mask[1, 2:6, 2:6] = 1.0

    reference, debug_strip = node.build(image, mask, 2, 8, 32, 0.5)

    assert reference.shape[0] == 2
    assert reference.shape[1] == reference.shape[2]
    assert 8 <= reference.shape[1] <= 32
    assert float(reference[0, :, :, 0].mean().item()) > 0.0
    assert float(reference[1, :, :, 1].mean().item()) > 0.0
    assert debug_strip.shape == (2, 16, 16, 4)
    assert float(debug_strip[0, 0, 0, 3].item()) == 0.0
    assert float(debug_strip[0, 4, 3, 3].item()) == 1.0


def test_color_transfer_ref_from_mask_band_rejects_empty_mask() -> None:
    node = ColorTransferRefFromMaskBandNode()
    image = torch.zeros((1, 16, 16, 3), dtype=torch.float32)
    mask = torch.zeros((1, 16, 16), dtype=torch.float32)

    with pytest.raises(ValueError, match="Mask has no active pixels"):
        node.build(image, mask, 4, 8, 32, 0.5)

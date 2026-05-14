from __future__ import annotations

import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT.parent))

from comfy_custom_nodes_repo.nodes.zero_drift_inpaint_crop_stitch_node import (
    ZeroDriftInpaintCropNode,
    ZeroDriftInpaintStitchNode,
)


def _gradient_image(height: int, width: int) -> torch.Tensor:
    y = torch.linspace(0.0, 1.0, height).view(1, height, 1, 1).expand(1, height, width, 1)
    x = torch.linspace(0.0, 1.0, width).view(1, 1, width, 1).expand(1, height, width, 1)
    return torch.cat([x, y, (x + y) * 0.5], dim=-1).float()


def _crop(
    image: torch.Tensor,
    mask: torch.Tensor | None,
    optional_context_mask: torch.Tensor | None = None,
    *,
    mask_expand_pixels: int = 0,
    mask_blend_pixels: int = 0,
    context_from_mask_extend_factor: float = 1.0,
) -> tuple[dict, torch.Tensor, torch.Tensor]:
    node = ZeroDriftInpaintCropNode()
    return node.inpaint_crop(
        image,
        "bilinear",
        "bicubic",
        mask_expand_pixels,
        mask_blend_pixels,
        context_from_mask_extend_factor,
        mask,
        optional_context_mask,
    )


def test_removed_ui_fields_are_not_exposed() -> None:
    required = ZeroDriftInpaintCropNode.INPUT_TYPES()["required"]
    removed_fields = {
        "preresize",
        "preresize_mode",
        "preresize_min_width",
        "preresize_min_height",
        "preresize_max_width",
        "preresize_max_height",
        "mask_fill_holes",
        "mask_invert",
        "mask_hipass_filter",
        "extend_for_outpainting",
        "extend_up_factor",
        "extend_down_factor",
        "extend_left_factor",
        "extend_right_factor",
        "output_resize_to_target_size",
        "output_target_width",
        "output_target_height",
        "output_padding",
        "device_mode",
    }
    assert removed_fields.isdisjoint(required.keys())


def test_round_trip_is_exact_without_blend() -> None:
    image = _gradient_image(101, 157)
    mask = torch.zeros((1, 101, 157), dtype=torch.float32)
    mask[:, 20:61, 40:96] = 1.0

    stitch = ZeroDriftInpaintStitchNode()
    stitcher, cropped_image, _cropped_mask = _crop(
        image,
        mask,
        context_from_mask_extend_factor=1.0,
    )
    restored, = stitch.inpaint_stitch(stitcher, cropped_image)
    assert torch.equal(restored, image)
    assert cropped_image.shape[1:3] == (41, 56)


def test_blend_mask_never_changes_pixels_outside_selection() -> None:
    image = _gradient_image(101, 157)
    mask = torch.zeros((1, 101, 157), dtype=torch.float32)
    mask[:, 20:61, 40:96] = 1.0

    stitch = ZeroDriftInpaintStitchNode()
    stitcher, cropped_image, _cropped_mask = _crop(
        image,
        mask,
        mask_blend_pixels=32,
        context_from_mask_extend_factor=1.2,
    )
    restored, = stitch.inpaint_stitch(stitcher, cropped_image)
    diff = (restored - image).abs()
    outside = diff * (1.0 - mask.unsqueeze(-1))
    assert float(outside.max()) == 0.0


def test_mask_expand_pixels_enlarges_crop_geometry() -> None:
    image = _gradient_image(101, 157)
    mask = torch.zeros((1, 101, 157), dtype=torch.float32)
    mask[:, 20:61, 40:96] = 1.0

    stitcher_base, _image_a, _mask_a = _crop(
        image,
        mask,
    )
    stitcher_expanded, _image_b, _mask_b = _crop(
        image,
        mask,
        mask_expand_pixels=3,
    )

    assert stitcher_expanded["cropped_to_canvas_w"][0] > stitcher_base["cropped_to_canvas_w"][0]
    assert stitcher_expanded["cropped_to_canvas_h"][0] > stitcher_base["cropped_to_canvas_h"][0]


def test_single_stitcher_can_drive_mask_batch() -> None:
    image = _gradient_image(64, 96)
    mask = torch.zeros((1, 64, 96), dtype=torch.float32)
    mask[:, 16:48, 24:72] = 1.0
    stitch = ZeroDriftInpaintStitchNode()

    stitcher, cropped_image, _cropped_mask = _crop(
        image,
        mask,
    )

    duplicated = torch.cat([cropped_image, cropped_image], dim=0)
    restored, = stitch.inpaint_stitch(stitcher, duplicated)
    assert restored.shape[0] == 2
    assert torch.equal(restored[0], restored[1])


def test_empty_mask_falls_back_to_full_image_without_drift() -> None:
    image = _gradient_image(73, 111)
    mask = torch.zeros((1, 73, 111), dtype=torch.float32)
    stitch = ZeroDriftInpaintStitchNode()

    stitcher, cropped_image, cropped_mask = _crop(
        image,
        mask,
    )
    restored, = stitch.inpaint_stitch(stitcher, cropped_image)
    assert torch.equal(restored, image)
    assert int(cropped_mask.sum().item()) == 0


def test_optional_context_mask_enlarges_crop_but_round_trip_stays_exact() -> None:
    image = _gradient_image(128, 160)
    mask = torch.zeros((1, 128, 160), dtype=torch.float32)
    context = torch.zeros((1, 128, 160), dtype=torch.float32)
    mask[:, 40:72, 48:80] = 1.0
    context[:, 20:100, 24:120] = 1.0
    stitch = ZeroDriftInpaintStitchNode()

    stitcher, cropped_image, _cropped_mask = _crop(
        image,
        mask,
        optional_context_mask=context,
    )
    restored, = stitch.inpaint_stitch(stitcher, cropped_image)
    assert torch.equal(restored, image)
    assert stitcher["cropped_to_canvas_w"][0] > 32
    assert stitcher["cropped_to_canvas_h"][0] > 32


def test_large_blend_radius_does_not_crash_on_small_crop() -> None:
    image = _gradient_image(64, 96)
    mask = torch.zeros((1, 64, 96), dtype=torch.float32)
    mask[:, 20:24, 40:44] = 1.0
    stitch = ZeroDriftInpaintStitchNode()

    stitcher, cropped_image, _cropped_mask = _crop(
        image,
        mask,
        mask_blend_pixels=48,
    )
    restored, = stitch.inpaint_stitch(stitcher, cropped_image)
    outside = ((restored - image).abs() * (1.0 - mask.unsqueeze(-1))).max().item()
    assert outside == 0.0

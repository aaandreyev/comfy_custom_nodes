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


def test_round_trip_is_exact_without_resize_or_blend() -> None:
    image = _gradient_image(101, 157)
    mask = torch.zeros((1, 101, 157), dtype=torch.float32)
    mask[:, 20:61, 40:96] = 1.0

    crop = ZeroDriftInpaintCropNode()
    stitch = ZeroDriftInpaintStitchNode()
    stitcher, cropped_image, _cropped_mask = crop.inpaint_crop(
        image,
        "bilinear",
        "bicubic",
        False,
        "ensure minimum resolution",
        1024,
        1024,
        4096,
        4096,
        False,
        1.0,
        1.0,
        1.0,
        1.0,
        0.1,
        True,
        0,
        False,
        0,
        1.2,
        False,
        512,
        512,
        "0",
        "cpu (deterministic)",
        mask,
        None,
    )
    restored, = stitch.inpaint_stitch(stitcher, cropped_image)
    assert torch.equal(restored, image)


def test_blend_mask_never_changes_pixels_outside_selection() -> None:
    image = _gradient_image(101, 157)
    mask = torch.zeros((1, 101, 157), dtype=torch.float32)
    mask[:, 20:61, 40:96] = 1.0

    crop = ZeroDriftInpaintCropNode()
    stitch = ZeroDriftInpaintStitchNode()
    stitcher, cropped_image, _cropped_mask = crop.inpaint_crop(
        image,
        "bilinear",
        "bicubic",
        False,
        "ensure minimum resolution",
        1024,
        1024,
        4096,
        4096,
        False,
        1.0,
        1.0,
        1.0,
        1.0,
        0.1,
        True,
        0,
        False,
        32,
        1.2,
        True,
        512,
        512,
        "32",
        "cpu (deterministic)",
        mask,
        None,
    )
    restored, = stitch.inpaint_stitch(stitcher, cropped_image)
    diff = (restored - image).abs()
    outside = diff * (1.0 - mask.unsqueeze(-1))
    assert float(outside.max()) == 0.0


def test_output_padding_does_not_change_crop_geometry() -> None:
    image = _gradient_image(101, 157)
    mask = torch.zeros((1, 101, 157), dtype=torch.float32)
    mask[:, 20:61, 40:96] = 1.0
    crop = ZeroDriftInpaintCropNode()

    stitcher_no_pad, _image_a, _mask_a = crop.inpaint_crop(
        image,
        "bilinear",
        "bicubic",
        False,
        "ensure minimum resolution",
        1024,
        1024,
        4096,
        4096,
        False,
        1.0,
        1.0,
        1.0,
        1.0,
        0.1,
        True,
        0,
        False,
        0,
        1.2,
        True,
        513,
        512,
        "0",
        "cpu (deterministic)",
        mask,
        None,
    )
    stitcher_pad, _image_b, _mask_b = crop.inpaint_crop(
        image,
        "bilinear",
        "bicubic",
        False,
        "ensure minimum resolution",
        1024,
        1024,
        4096,
        4096,
        False,
        1.0,
        1.0,
        1.0,
        1.0,
        0.1,
        True,
        0,
        False,
        0,
        1.2,
        True,
        513,
        512,
        "32",
        "cpu (deterministic)",
        mask,
        None,
    )

    keys = [
        "cropped_to_canvas_x",
        "cropped_to_canvas_y",
        "cropped_to_canvas_w",
        "cropped_to_canvas_h",
    ]
    for key in keys:
        assert stitcher_no_pad[key] == stitcher_pad[key]


def test_single_stitcher_can_drive_mask_batch() -> None:
    image = _gradient_image(64, 96)
    mask = torch.zeros((1, 64, 96), dtype=torch.float32)
    mask[:, 16:48, 24:72] = 1.0
    crop = ZeroDriftInpaintCropNode()
    stitch = ZeroDriftInpaintStitchNode()
    stitcher, cropped_image, _cropped_mask = crop.inpaint_crop(
        image,
        "bilinear",
        "bicubic",
        False,
        "ensure minimum resolution",
        1024,
        1024,
        4096,
        4096,
        False,
        1.0,
        1.0,
        1.0,
        1.0,
        0.1,
        True,
        0,
        False,
        0,
        1.0,
        True,
        256,
        256,
        "0",
        "cpu (deterministic)",
        mask,
        None,
    )

    duplicated = torch.cat([cropped_image, cropped_image], dim=0)
    restored, = stitch.inpaint_stitch(stitcher, duplicated)
    assert restored.shape[0] == 2
    assert torch.equal(restored[0], restored[1])


def test_empty_mask_falls_back_to_full_image_without_drift() -> None:
    image = _gradient_image(73, 111)
    mask = torch.zeros((1, 73, 111), dtype=torch.float32)
    crop = ZeroDriftInpaintCropNode()
    stitch = ZeroDriftInpaintStitchNode()

    stitcher, cropped_image, cropped_mask = crop.inpaint_crop(
        image,
        "bilinear",
        "bicubic",
        False,
        "ensure minimum resolution",
        1024,
        1024,
        4096,
        4096,
        False,
        1.0,
        1.0,
        1.0,
        1.0,
        0.1,
        True,
        0,
        False,
        0,
        1.0,
        False,
        512,
        512,
        "0",
        "cpu (deterministic)",
        mask,
        None,
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

    crop = ZeroDriftInpaintCropNode()
    stitch = ZeroDriftInpaintStitchNode()
    stitcher, cropped_image, _cropped_mask = crop.inpaint_crop(
        image,
        "bilinear",
        "bicubic",
        False,
        "ensure minimum resolution",
        1024,
        1024,
        4096,
        4096,
        False,
        1.0,
        1.0,
        1.0,
        1.0,
        0.1,
        True,
        0,
        False,
        0,
        1.0,
        False,
        512,
        512,
        "0",
        "cpu (deterministic)",
        mask,
        context,
    )
    restored, = stitch.inpaint_stitch(stitcher, cropped_image)
    assert torch.equal(restored, image)
    assert stitcher["cropped_to_canvas_w"][0] > 32
    assert stitcher["cropped_to_canvas_h"][0] > 32


def test_outpaint_extend_keeps_original_region_exact_on_identity_round_trip() -> None:
    image = _gradient_image(96, 144)
    mask = torch.zeros((1, 96, 144), dtype=torch.float32)
    mask[:, :, 120:] = 1.0

    crop = ZeroDriftInpaintCropNode()
    stitch = ZeroDriftInpaintStitchNode()
    stitcher, cropped_image, _cropped_mask = crop.inpaint_crop(
        image,
        "bilinear",
        "bicubic",
        False,
        "ensure minimum resolution",
        1024,
        1024,
        4096,
        4096,
        True,
        1.0,
        1.0,
        1.0,
        2.0,
        0.0,
        False,
        0,
        False,
        0,
        1.15,
        False,
        1536,
        512,
        "0",
        "cpu (deterministic)",
        mask,
        None,
    )
    restored, = stitch.inpaint_stitch(stitcher, cropped_image)
    assert torch.equal(restored, image)


def test_preresize_keeps_single_pixel_mask_discrete() -> None:
    image = _gradient_image(16, 16)
    mask = torch.zeros((1, 16, 16), dtype=torch.float32)
    mask[:, 8, 8] = 1.0
    crop = ZeroDriftInpaintCropNode()

    _stitcher, _cropped_image, cropped_mask = crop.inpaint_crop(
        image,
        "bilinear",
        "bicubic",
        True,
        "ensure minimum resolution",
        64,
        64,
        4096,
        4096,
        False,
        1.0,
        1.0,
        1.0,
        1.0,
        0.1,
        False,
        0,
        False,
        0,
        1.0,
        False,
        64,
        64,
        "0",
        "cpu (deterministic)",
        mask,
        None,
    )
    unique = set(torch.unique(cropped_mask).tolist())
    assert unique.issubset({0.0, 1.0})
    assert int(cropped_mask.sum().item()) >= 1


def test_large_blend_radius_does_not_crash_on_small_crop() -> None:
    image = _gradient_image(64, 96)
    mask = torch.zeros((1, 64, 96), dtype=torch.float32)
    mask[:, 20:24, 40:44] = 1.0
    crop = ZeroDriftInpaintCropNode()
    stitch = ZeroDriftInpaintStitchNode()

    stitcher, cropped_image, _cropped_mask = crop.inpaint_crop(
        image,
        "bilinear",
        "bicubic",
        False,
        "ensure minimum resolution",
        1024,
        1024,
        4096,
        4096,
        False,
        1.0,
        1.0,
        1.0,
        1.0,
        0.1,
        True,
        0,
        False,
        48,
        1.0,
        True,
        64,
        64,
        "32",
        "cpu (deterministic)",
        mask,
        None,
    )
    restored, = stitch.inpaint_stitch(stitcher, cropped_image)
    outside = ((restored - image).abs() * (1.0 - mask.unsqueeze(-1))).max().item()
    assert outside == 0.0

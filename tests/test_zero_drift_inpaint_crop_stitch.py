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


def _assert_rgb_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Allow small drift from resize crop ↔ canvas round-trip when VAE alignment is on."""
    torch.testing.assert_close(actual, expected, rtol=5e-4, atol=5e-4)


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
    vae_alignment_multiple_of_8: bool = True,
) -> tuple[dict, torch.Tensor, torch.Tensor]:
    node = ZeroDriftInpaintCropNode()
    return node.inpaint_crop(
        image,
        "bilinear",
        "bicubic",
        mask_expand_pixels,
        mask_blend_pixels,
        context_from_mask_extend_factor,
        vae_alignment_multiple_of_8,
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


def test_vae_alignment_disabled_keeps_natural_crop_shape_and_exact_round_trip() -> None:
    image = _gradient_image(101, 157)
    mask = torch.zeros((1, 101, 157), dtype=torch.float32)
    mask[:, 20:61, 40:96] = 1.0

    stitch = ZeroDriftInpaintStitchNode()
    stitcher, cropped_image, _cropped_mask = _crop(
        image,
        mask,
        context_from_mask_extend_factor=1.0,
        vae_alignment_multiple_of_8=False,
    )
    restored, = stitch.inpaint_stitch(stitcher, cropped_image)
    assert torch.equal(restored, image)
    assert cropped_image.shape[1:3] == (41, 56)


def test_round_trip_is_exact_without_blend() -> None:
    image = _gradient_image(101, 157)
    mask = torch.zeros((1, 101, 157), dtype=torch.float32)
    mask[:, 20:68, 40:104] = 1.0

    stitch = ZeroDriftInpaintStitchNode()
    stitcher, cropped_image, _cropped_mask = _crop(
        image,
        mask,
        context_from_mask_extend_factor=1.0,
    )
    restored, = stitch.inpaint_stitch(stitcher, cropped_image)
    _assert_rgb_close(restored, image)
    assert cropped_image.shape[1] % 16 == 0 and cropped_image.shape[2] % 16 == 0
    assert cropped_image.shape[1:3] == (48, 64)


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
    assert float(outside.max()) <= 1e-4


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
    _assert_rgb_close(restored, image)
    assert int(cropped_mask.sum().item()) == 0
    assert cropped_image.shape[1] % 8 == 0 and cropped_image.shape[2] % 8 == 0


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
    _assert_rgb_close(restored, image)
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


def _vae_center_crop(image: torch.Tensor, multiple: int) -> torch.Tensor:
    """comfy.sd.VAE.vae_encode_crop_pixels: symmetric crop of H,W down to a multiple."""
    h = image.shape[1] // multiple * multiple
    w = image.shape[2] // multiple * multiple
    y0 = (image.shape[1] % multiple) // 2
    x0 = (image.shape[2] % multiple) // 2
    return image[:, y0 : y0 + h, x0 : x0 + w, :]


def test_default_alignment_matches_flux2_vae_multiple_of_16() -> None:
    image = _gradient_image(320, 320)
    mask = torch.zeros(1, 320, 320)
    mask[:, 10:186, 10:258] = 1.0
    _, cropped_image, _ = _crop(image, mask)
    assert cropped_image.shape[1] % 16 == 0
    assert cropped_image.shape[2] % 16 == 0
    assert _vae_center_crop(cropped_image, 16).shape == cropped_image.shape


def test_stitch_accepts_vae16_center_cropped_result_from_legacy_multiple_of_8_crop() -> None:
    image = _gradient_image(320, 320)
    mask = torch.zeros(1, 320, 320)
    mask[:, 10:186, 10:258] = 1.0
    node = ZeroDriftInpaintCropNode()
    stitcher, cropped_image, _ = node.inpaint_crop(
        image,
        "bilinear",
        "bicubic",
        0,
        0,
        1.0,
        True,
        mask,
        None,
        vae_size_multiple=8,
    )
    assert cropped_image.shape[1:3] == (176, 248)
    decoded = _vae_center_crop(cropped_image, 16)
    assert decoded.shape[1:3] == (176, 240)
    inpainted = torch.full_like(decoded, 0.5)
    result = ZeroDriftInpaintStitchNode().inpaint_stitch(stitcher, inpainted)[0]

    assert result.shape == image.shape
    outside = mask[0] <= 0.5
    torch.testing.assert_close(result[:, outside, :], image[:, outside, :])
    torch.testing.assert_close(
        result[:, 20:170, 30:230, :], torch.full_like(result[:, 20:170, 30:230, :], 0.5)
    )
    left_margin = image[:, 10:186, 10:14, :]
    torch.testing.assert_close(result[:, 10:186, 10:14, :], left_margin)


def test_stitch_still_rejects_non_vae_shaped_mismatch() -> None:
    image = _gradient_image(320, 320)
    mask = torch.zeros(1, 320, 320)
    mask[:, 10:186, 10:258] = 1.0
    node = ZeroDriftInpaintCropNode()
    stitcher, cropped_image, _ = node.inpaint_crop(
        image, "bilinear", "bicubic", 0, 0, 1.0, True, mask, None, vae_size_multiple=8
    )
    bad = cropped_image[:, :, :-1, :]
    try:
        ZeroDriftInpaintStitchNode().inpaint_stitch(stitcher, bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("expected shape-mismatch AssertionError for non-VAE-shaped input")


def _crop_bucketed(image, mask, *, bucket=128, bucket_min=512, **kw):
    node = ZeroDriftInpaintCropNode()
    return node.inpaint_crop(
        image, "bilinear", "bicubic", 0, 0,
        kw.pop("context_from_mask_extend_factor", 1.25),
        True, mask, None,
        vae_size_multiple=16, size_bucket_px=bucket, bucket_min_px=bucket_min,
    )


def test_bucketing_lands_on_grid_and_keeps_real_pixels() -> None:
    image = _gradient_image(2048, 2048)
    mask = torch.zeros(1, 2048, 2048)
    mask[:, 900:990, 800:1000] = 1.0
    stitcher, cropped, _ = _crop_bucketed(image, mask)
    h, w = cropped.shape[1:3]
    assert (h, w) == (512, 512)
    x = stitcher["cropped_to_canvas_x"][0]
    y = stitcher["cropped_to_canvas_y"][0]
    torch.testing.assert_close(cropped, image[:, y : y + h, x : x + w, :])


def test_bucketing_rounds_up_to_next_grid_step() -> None:
    image = _gradient_image(2048, 2048)
    mask = torch.zeros(1, 2048, 2048)
    mask[:, 500:1100, 400:950] = 1.0
    _, cropped, _ = _crop_bucketed(image, mask)
    h, w = cropped.shape[1:3]
    assert h % 128 == 0 and w % 128 == 0
    assert h >= 600 * 1.25 and w >= 550 * 1.25
    assert (h, w) == (768, 768)


def test_bucketing_slides_window_at_canvas_corner() -> None:
    image = _gradient_image(2048, 2048)
    mask = torch.zeros(1, 2048, 2048)
    mask[:, 0:80, 0:80] = 1.0
    stitcher, cropped, _ = _crop_bucketed(image, mask)
    assert cropped.shape[1:3] == (512, 512)
    assert stitcher["cropped_to_canvas_x"][0] == 0
    assert stitcher["cropped_to_canvas_y"][0] == 0


def test_bucketing_capped_by_small_canvas() -> None:
    image = _gradient_image(400, 400)
    mask = torch.zeros(1, 400, 400)
    mask[:, 100:300, 100:300] = 1.0
    _, cropped, _ = _crop_bucketed(image, mask)
    assert cropped.shape[1:3] == (400, 400)


def test_bucketing_round_trip_stays_exact() -> None:
    image = _gradient_image(1024, 1024)
    mask = torch.zeros(1, 1024, 1024)
    mask[:, 300:520, 350:600] = 1.0
    stitcher, cropped, _ = _crop_bucketed(image, mask)
    restored, = ZeroDriftInpaintStitchNode().inpaint_stitch(stitcher, cropped)
    torch.testing.assert_close(restored, image)


def test_bucketing_off_keeps_legacy_crop_box() -> None:
    image = _gradient_image(1024, 1024)
    mask = torch.zeros(1, 1024, 1024)
    mask[:, 300:520, 350:600] = 1.0
    node = ZeroDriftInpaintCropNode()
    _, cropped_off, _ = node.inpaint_crop(
        image, "bilinear", "bicubic", 0, 0, 1.25, True, mask, None,
        vae_size_multiple=16, size_bucket_px=0,
    )
    _, cropped_legacy, _ = node.inpaint_crop(
        image, "bilinear", "bicubic", 0, 0, 1.25, True, mask, None,
        vae_size_multiple=16,
    )
    torch.testing.assert_close(cropped_off, cropped_legacy)


def test_bucketing_batch_shares_one_bucketed_box() -> None:
    image = _gradient_image(1024, 1024).expand(2, -1, -1, -1).clone()
    mask = torch.zeros(2, 1024, 1024)
    mask[0, 300:420, 350:500] = 1.0
    mask[1, 500:640, 420:560] = 1.0
    _, cropped, _ = _crop_bucketed(image, mask)
    assert cropped.shape[0] == 2
    assert cropped.shape[1] % 128 == 0 and cropped.shape[2] % 128 == 0


def test_bucketing_downscales_oversized_crop_to_max_side() -> None:
    image = _gradient_image(2048, 2048)
    mask = torch.zeros(1, 2048, 2048)
    mask[:, 200:1100, 100:1800] = 1.0
    stitcher, cropped, _ = _crop_bucketed(image, mask)
    h, w = cropped.shape[1:3]
    assert w == 1536
    assert h == 896
    restored, = ZeroDriftInpaintStitchNode().inpaint_stitch(stitcher, cropped)
    outside = ((restored - image).abs() * (1.0 - mask.unsqueeze(-1))).max().item()
    assert outside <= 1e-4


def test_bucketing_four_tile_corner_maps_to_1536_square() -> None:
    image = _gradient_image(2048, 2048)
    mask = torch.zeros(1, 2048, 2048)
    mask[:, 124:1924, 124:1924] = 1.0
    _, cropped, _ = _crop_bucketed(image, mask)
    assert cropped.shape[1:3] == (1536, 1536)


def test_batched_stitch_accepts_vae_center_cropped_results() -> None:
    image = _gradient_image(320, 320).expand(2, -1, -1, -1).clone()
    mask = torch.zeros(2, 320, 320)
    mask[:, 10:186, 10:258] = 1.0
    node = ZeroDriftInpaintCropNode()
    stitcher, cropped, _ = node.inpaint_crop(
        image, "bilinear", "bicubic", 0, 0, 1.0, True, mask, None, vae_size_multiple=8
    )
    decoded = _vae_center_crop(cropped, 16)
    assert decoded.shape[0] == 2 and decoded.shape[1:3] == (176, 240)
    inpainted = torch.full_like(decoded, 0.5)
    result = ZeroDriftInpaintStitchNode().inpaint_stitch(stitcher, inpainted)[0]
    assert result.shape[0] == 2
    for i in range(2):
        outside = mask[i] <= 0.5
        torch.testing.assert_close(result[i][outside], image[i][outside])

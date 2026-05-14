"""
Matrix / property-style tests for Zero Drift crop ↔ stitch.

Notes:
- Identity round-trip with ``align_crop_spatial_multiple_of_8=True`` resamples inside the
  inpainted region; smooth synthetic RGB patterns keep global error bounded (see
  ``_assert_aligned_smooth_identity``). Uniform noise is checked via outside-mask invariants +
  divisibility instead of full-frame closeness.
- ``run_zero_drift_crop`` stacks crops across batch **only when every slice has the same
  cropped H×W** after alignment; batch tests construct masks with identical bbox extents.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT.parent))

from comfy_custom_nodes_repo.runtime.infer.zero_drift_inpaint_crop import (
    run_zero_drift_crop,
    stitch_zero_drift_result,
)


def _nearest_multiple_of_8(x: int) -> int:
    x = int(x)
    if x <= 0:
        return 8
    return max(8, int(round(x / 8.0)) * 8)


def _assert_aligned_smooth_identity(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Identity crop→×8→stitch resamples inside mask; bound total deviation for smooth RGB."""
    torch.testing.assert_close(actual, expected, rtol=5e-2, atol=5e-2)


def _smooth_rgb(batch: int, height: int, width: int) -> torch.Tensor:
    """Low-frequency gradient RGB — mild interpolation drift under crop ×8 resize."""
    y = torch.linspace(0.0, 1.0, height, dtype=torch.float32).view(1, height, 1, 1).expand(batch, height, width, 1)
    x = torch.linspace(0.0, 1.0, width, dtype=torch.float32).view(1, 1, width, 1).expand(batch, height, width, 1)
    return torch.cat([x, y, (x + y) * 0.5], dim=-1)


def _uniform_noise(batch: int, height: int, width: int, generator: torch.Generator) -> torch.Tensor:
    return torch.rand(batch, height, width, 3, generator=generator, dtype=torch.float32)


def _random_rect_mask(batch: int, height: int, width: int, generator: torch.Generator) -> torch.Tensor:
    m = torch.zeros(batch, height, width, dtype=torch.float32)
    for b in range(batch):
        hh = int(torch.randint(1, height + 1, (1,), generator=generator).item())
        ww = int(torch.randint(1, width + 1, (1,), generator=generator).item())
        y0 = int(torch.randint(0, max(1, height - hh + 1), (1,), generator=generator).item())
        x0 = int(torch.randint(0, max(1, width - ww + 1), (1,), generator=generator).item())
        m[b, y0 : y0 + hh, x0 : x0 + ww] = 1.0
    return m


def _scatter_mask(batch: int, height: int, width: int, n_pixels: int, generator: torch.Generator) -> torch.Tensor:
    m = torch.zeros(batch, height, width, dtype=torch.float32)
    for b in range(batch):
        flat_idx = torch.randperm(height * width, generator=generator)[: min(n_pixels, height * width)]
        fy = flat_idx // width
        fx = flat_idx % width
        m[b, fy, fx] = 1.0
    return m


def _crop_stitch(
    image: torch.Tensor,
    mask: torch.Tensor | None,
    *,
    optional_context_mask: torch.Tensor | None = None,
    downscale_algorithm: str = "bilinear",
    upscale_algorithm: str = "bicubic",
    mask_expand_pixels: int = 0,
    mask_blend_pixels: int = 0,
    context_from_mask_extend_factor: float = 1.0,
    align_crop_spatial_multiple_of_8: bool = True,
) -> tuple[torch.Tensor, dict, torch.Tensor, torch.Tensor]:
    stitcher, cropped_image, cropped_mask = run_zero_drift_crop(
        image=image,
        downscale_algorithm=downscale_algorithm,
        upscale_algorithm=upscale_algorithm,
        mask_expand_pixels=mask_expand_pixels,
        mask_blend_pixels=mask_blend_pixels,
        context_from_mask_extend_factor=context_from_mask_extend_factor,
        mask=mask,
        optional_context_mask=optional_context_mask,
        align_crop_spatial_multiple_of_8=align_crop_spatial_multiple_of_8,
    )
    restored = stitch_zero_drift_result(stitcher, cropped_image.clone())
    return restored, stitcher, cropped_image, cropped_mask


def _natural_crop_wh(stitcher: dict, idx: int = 0) -> tuple[int, int]:
    return int(stitcher["cropped_to_canvas_w"][idx]), int(stitcher["cropped_to_canvas_h"][idx])


def _assert_align_hw_and_optional_noise_outside_mask(
    restored: torch.Tensor,
    image: torch.Tensor,
    mask: torch.Tensor,
    *,
    outside_max: float,
) -> None:
    """For noisy inputs + ×8 alignment: geometry ok + strictly untouched outside binary mask."""
    assert restored.shape == image.shape
    diff = (restored - image).abs()
    outside = diff * (1.0 - mask.unsqueeze(-1))
    assert float(outside.max()) <= outside_max


@pytest.mark.parametrize(
    ("height", "width"),
    [
        (8, 8),
        (9, 15),
        (17, 33),
        (41, 56),
        (63, 127),
        (73, 111),
        (99, 101),
        (128, 160),
        (8, 25),
        (25, 8),
    ],
)
@pytest.mark.parametrize("align_vae", [True, False])
def test_identity_roundtrip_rect_mask_various_canvas_sizes(height: int, width: int, align_vae: bool) -> None:
    gen = torch.Generator().manual_seed(height * 10007 + width * 131 + int(align_vae))
    mask = _random_rect_mask(1, height, width, gen)

    if align_vae:
        image = _smooth_rgb(1, height, width)
    else:
        image = _uniform_noise(1, height, width, gen)

    restored, stitcher, cropped_image, cropped_mask = _crop_stitch(
        image,
        mask,
        align_crop_spatial_multiple_of_8=align_vae,
    )

    assert restored.shape == image.shape
    nat_w, nat_h = _natural_crop_wh(stitcher, 0)
    ew, eh = _nearest_multiple_of_8(nat_w), _nearest_multiple_of_8(nat_h)

    if align_vae:
        assert cropped_image.shape[1] % 8 == 0 and cropped_image.shape[2] % 8 == 0
        assert cropped_mask.shape[1:] == cropped_image.shape[1:3]
        assert cropped_image.shape[2] == ew and cropped_image.shape[1] == eh
        _assert_aligned_smooth_identity(restored, image)
    else:
        assert cropped_image.shape[2] == nat_w and cropped_image.shape[1] == nat_h
        torch.testing.assert_close(restored, image)


@pytest.mark.parametrize("seed", list(range(24)))
@pytest.mark.parametrize("align_vae", [True, False])
def test_identity_roundtrip_random_sizes_and_masks(seed: int, align_vae: bool) -> None:
    gen = torch.Generator().manual_seed(seed + (9026 if align_vae else 0))
    height = int(torch.randint(8, 96, (1,), generator=gen).item())
    width = int(torch.randint(8, 96, (1,), generator=gen).item())
    pattern = int(torch.randint(0, 3, (1,), generator=gen).item())

    if align_vae:
        image = _smooth_rgb(1, height, width)
    else:
        image = _uniform_noise(1, height, width, gen)

    if pattern == 0:
        mask = _random_rect_mask(1, height, width, gen)
    elif pattern == 1:
        mask = _scatter_mask(1, height, width, max(4, height * width // 20), gen)
    else:
        mask = torch.ones(1, height, width, dtype=torch.float32)

    restored, stitcher, cropped_image, cropped_mask = _crop_stitch(
        image,
        mask,
        align_crop_spatial_multiple_of_8=align_vae,
    )

    assert restored.shape == image.shape
    nat_w, nat_h = _natural_crop_wh(stitcher, 0)

    if align_vae:
        assert cropped_image.shape[1] % 8 == 0 and cropped_image.shape[2] % 8 == 0
        assert cropped_image.shape == (
            1,
            _nearest_multiple_of_8(nat_h),
            _nearest_multiple_of_8(nat_w),
            3,
        )
        assert cropped_mask.shape[1:] == cropped_image.shape[1:3]
        if pattern == 2:
            torch.testing.assert_close(restored, image, rtol=1.5e-1, atol=1.5e-1)
        else:
            _assert_aligned_smooth_identity(restored, image)
    else:
        torch.testing.assert_close(restored, image)


@pytest.mark.parametrize("seed", list(range(16)))
def test_noise_image_alignment_preserves_outside_mask_only(seed: int) -> None:
    gen = torch.Generator().manual_seed(seed + 5555)
    height = int(torch.randint(16, 80, (1,), generator=gen).item())
    width = int(torch.randint(16, 80, (1,), generator=gen).item())
    image = _uniform_noise(1, height, width, gen)
    mask = _random_rect_mask(1, height, width, gen)

    restored, stitcher, cropped_image, _ = _crop_stitch(
        image,
        mask,
        align_crop_spatial_multiple_of_8=True,
    )

    assert cropped_image.shape[1] % 8 == 0 and cropped_image.shape[2] % 8 == 0
    nat_w, nat_h = _natural_crop_wh(stitcher, 0)
    assert cropped_image.shape[2] == _nearest_multiple_of_8(nat_w)
    assert cropped_image.shape[1] == _nearest_multiple_of_8(nat_h)
    _assert_align_hw_and_optional_noise_outside_mask(restored, image, mask, outside_max=5e-4)


@pytest.mark.parametrize(
    ("downscale_algorithm", "upscale_algorithm"),
    [
        ("nearest", "nearest"),
        ("bilinear", "bicubic"),
        ("bicubic", "bilinear"),
        ("box", "hamming"),
    ],
)
@pytest.mark.parametrize("align_vae", [True, False])
def test_resize_algorithm_matrix_roundtrip(downscale_algorithm: str, upscale_algorithm: str, align_vae: bool) -> None:
    height, width = 77, 89
    image = _smooth_rgb(1, height, width)
    mask = torch.zeros(1, height, width, dtype=torch.float32)
    mask[:, 30:65, 22:71] = 1.0

    restored, _, cropped_image, _ = _crop_stitch(
        image,
        mask,
        downscale_algorithm=downscale_algorithm,
        upscale_algorithm=upscale_algorithm,
        align_crop_spatial_multiple_of_8=align_vae,
    )

    assert restored.shape == image.shape
    if align_vae:
        assert cropped_image.shape[1] % 8 == 0 and cropped_image.shape[2] % 8 == 0
        _assert_aligned_smooth_identity(restored, image)
    else:
        torch.testing.assert_close(restored, image)


@pytest.mark.parametrize("mask_expand_pixels", [0, 2, 7])
@pytest.mark.parametrize("mask_blend_pixels", [0, 16, 48])
@pytest.mark.parametrize("context_from_mask_extend_factor", [1.0, 1.15, 1.35])
@pytest.mark.parametrize("align_vae", [True, False])
def test_combo_expand_blend_context_alignment(
    mask_expand_pixels: int,
    mask_blend_pixels: int,
    context_from_mask_extend_factor: float,
    align_vae: bool,
) -> None:
    height, width = 96, 112
    image = _smooth_rgb(1, height, width)
    mask = torch.zeros(1, height, width, dtype=torch.float32)
    mask[:, 40:72, 48:88] = 1.0
    context = torch.zeros(1, height, width, dtype=torch.float32)
    context[:, 28:84, 36:92] = 1.0

    restored, _, cropped_image, _ = _crop_stitch(
        image,
        mask,
        optional_context_mask=context,
        mask_expand_pixels=mask_expand_pixels,
        mask_blend_pixels=mask_blend_pixels,
        context_from_mask_extend_factor=context_from_mask_extend_factor,
        align_crop_spatial_multiple_of_8=align_vae,
    )

    assert restored.shape == image.shape
    if align_vae:
        assert cropped_image.shape[1] % 8 == 0 and cropped_image.shape[2] % 8 == 0
        _assert_aligned_smooth_identity(restored, image)

    diff = (restored - image).abs()
    outside = diff * (1.0 - mask.unsqueeze(-1))
    limit = 1e-3 if align_vae else 1e-6
    assert float(outside.max()) <= limit


@pytest.mark.parametrize("batch", [1, 2, 4])
@pytest.mark.parametrize("align_vae", [True, False])
def test_batch_identity_roundtrip_same_geometry(batch: int, align_vae: bool) -> None:
    height, width = 56, 72
    mask_single = torch.zeros(1, height, width, dtype=torch.float32)
    mask_single[:, 16:44, 20:52] = 1.0
    mask = mask_single.expand(batch, -1, -1).clone()

    if align_vae:
        image = _smooth_rgb(batch, height, width)
    else:
        gen = torch.Generator().manual_seed(batch * 809 + int(align_vae))
        image = _uniform_noise(batch, height, width, gen)

    restored, _, cropped_image, cropped_mask = _crop_stitch(image, mask, align_crop_spatial_multiple_of_8=align_vae)

    assert restored.shape == image.shape
    assert cropped_image.shape[0] == batch
    assert cropped_mask.shape[0] == batch
    if align_vae:
        assert cropped_image.shape[1] % 8 == 0 and cropped_image.shape[2] % 8 == 0
        _assert_aligned_smooth_identity(restored, image)
    else:
        torch.testing.assert_close(restored, image)


def test_batch_two_distinct_regions_same_crop_extent_stackable() -> None:
    """Two masks with identical rectangle size → identical aligned crop HW → batched stack."""
    height, width = 64, 80
    image = _smooth_rgb(2, height, width)
    mask = torch.zeros(2, height, width, dtype=torch.float32)
    mask[0, 10:26, 12:44] = 1.0  # 16×32
    mask[1, 38:54, 44:76] = 1.0  # 16×32

    restored, _, cropped_image, _ = _crop_stitch(image, mask, align_crop_spatial_multiple_of_8=True)

    assert restored.shape == image.shape
    assert cropped_image.shape[0] == 2
    _assert_aligned_smooth_identity(restored, image)


def test_empty_mask_full_canvas_roundtrip_aligned() -> None:
    height, width = 61, 83
    image = _smooth_rgb(1, height, width)
    mask = torch.zeros(1, height, width, dtype=torch.float32)

    restored, _, cropped_image, cropped_mask = _crop_stitch(image, mask, align_crop_spatial_multiple_of_8=True)

    assert restored.shape == image.shape
    ew = _nearest_multiple_of_8(width)
    eh = _nearest_multiple_of_8(height)
    assert cropped_image.shape == (1, eh, ew, 3)
    assert cropped_mask.sum().item() == 0
    _assert_aligned_smooth_identity(restored, image)


def test_single_pixel_mask_roundtrip_aligned() -> None:
    height, width = 40, 48
    image = _smooth_rgb(1, height, width)
    mask = torch.zeros(1, height, width, dtype=torch.float32)
    mask[:, height // 2, width // 2] = 1.0

    restored, _, cropped_image, _ = _crop_stitch(image, mask, align_crop_spatial_multiple_of_8=True)

    assert restored.shape == image.shape
    assert cropped_image.shape[1] % 8 == 0 and cropped_image.shape[2] % 8 == 0
    _assert_aligned_smooth_identity(restored, image)


def test_stitch_rejects_inpainted_wrong_hw() -> None:
    height, width = 40, 56
    image = torch.rand(1, height, width, 3)
    mask = torch.zeros(1, height, width)
    mask[:, 10:30, 12:44] = 1.0

    stitcher, cropped_image, _ = run_zero_drift_crop(
        image=image,
        downscale_algorithm="bilinear",
        upscale_algorithm="bicubic",
        mask_expand_pixels=0,
        mask_blend_pixels=0,
        context_from_mask_extend_factor=1.0,
        mask=mask,
        optional_context_mask=None,
        align_crop_spatial_multiple_of_8=True,
    )

    bad = cropped_image[:, :, :-1, :]
    with pytest.raises(AssertionError, match="does not match expected"):
        stitch_zero_drift_result(stitcher, bad)


def test_blend_tensor_shapes_match_for_uniform_batch_masks() -> None:
    """Same mask repeated → identical crops → stack succeeds; stitcher tensors consistent."""
    gen = torch.Generator().manual_seed(303)
    height, width = 52, 68
    batch = 3
    image = _smooth_rgb(batch, height, width)
    rect = torch.zeros(1, height, width, dtype=torch.float32)
    rect[:, 10:38, 14:54] = 1.0
    mask = rect.expand(batch, -1, -1).clone()

    stitcher, cropped_image, cropped_mask = run_zero_drift_crop(
        image=image + torch.rand(batch, height, width, 3, generator=gen) * 0.02,
        downscale_algorithm="bilinear",
        upscale_algorithm="bicubic",
        mask_expand_pixels=1,
        mask_blend_pixels=8,
        context_from_mask_extend_factor=1.05,
        mask=mask,
        optional_context_mask=None,
        align_crop_spatial_multiple_of_8=True,
    )

    assert len(stitcher["canvas_image"]) == batch
    ow = stitcher["output_content_w"][0]
    oh = stitcher["output_content_h"][0]
    assert cropped_image.shape == (batch, oh, ow, 3)
    assert cropped_mask.shape == (batch, oh, ow)
    for i in range(batch):
        assert stitcher["output_content_w"][i] == ow
        assert stitcher["output_content_h"][i] == oh
        bm = stitcher["blend_mask_output"][i]
        cs = stitcher["crop_support_canvas"][i]
        assert bm.shape == (1, oh, ow)
        assert cs.shape == (1, oh, ow)


def test_constant_color_roundtrip_exact_when_alignment_off_small_crop_divisible() -> None:
    height, width = 64, 64
    image = torch.full((1, height, width, 3), 0.375, dtype=torch.float32)
    mask = torch.zeros(1, height, width)
    mask[:, 16:48, 16:48] = 1.0

    restored, _, _, _ = _crop_stitch(image, mask, align_crop_spatial_multiple_of_8=False)
    assert torch.equal(restored, image)


def test_optional_context_binary_union_changes_bbox() -> None:
    height, width = 48, 64
    image = _smooth_rgb(1, height, width)
    mask = torch.zeros(1, height, width)
    mask[:, 10:22, 10:26] = 1.0
    ctx = torch.zeros(1, height, width)
    ctx[:, 20:40, 28:52] = 1.0

    _, stitcher_sel, _, _ = _crop_stitch(image, mask, optional_context_mask=torch.zeros_like(mask))
    _, stitcher_ctx, _, _ = _crop_stitch(image, mask, optional_context_mask=ctx)

    area_sel = int(stitcher_sel["cropped_to_canvas_w"][0]) * int(stitcher_sel["cropped_to_canvas_h"][0])
    area_ctx = int(stitcher_ctx["cropped_to_canvas_w"][0]) * int(stitcher_ctx["cropped_to_canvas_h"][0])
    assert area_ctx >= area_sel

    restored, _, _, _ = _crop_stitch(
        image,
        mask,
        optional_context_mask=ctx,
        align_crop_spatial_multiple_of_8=True,
    )
    _assert_aligned_smooth_identity(restored, image)


@pytest.mark.parametrize(
    "algo_ds,algo_us",
    [
        ("nearest", "bicubic"),
        ("lanczos", "box"),
    ],
)
def test_crop_stitch_via_nodes_matches_runtime(algo_ds: str, algo_us: str) -> None:
    from comfy_custom_nodes_repo.nodes.zero_drift_inpaint_crop_stitch_node import (
        ZeroDriftInpaintCropNode,
        ZeroDriftInpaintStitchNode,
    )

    height, width = 55, 71
    image = _smooth_rgb(1, height, width)
    mask = torch.zeros(1, height, width)
    mask[:, 18:42, 22:58] = 1.0

    crop_node = ZeroDriftInpaintCropNode()
    stitch_node = ZeroDriftInpaintStitchNode()
    stitcher_n, cropped_n, _ = crop_node.inpaint_crop(
        image,
        algo_ds,
        algo_us,
        0,
        0,
        1.0,
        True,
        mask,
        None,
    )
    restored_n, = stitch_node.inpaint_stitch(stitcher_n, cropped_n.clone())

    restored_r, _, _, _ = _crop_stitch(
        image,
        mask,
        downscale_algorithm=algo_ds,
        upscale_algorithm=algo_us,
        align_crop_spatial_multiple_of_8=True,
    )

    torch.testing.assert_close(restored_n, restored_r)


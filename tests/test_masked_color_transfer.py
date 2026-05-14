"""Smoke tests for masked color transfer (requires kornia)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("kornia")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT.parent))

from comfy_custom_nodes_repo.runtime.infer.color_transfer import color_transfer_images


@pytest.mark.parametrize("method", ["reinhard_lab", "mkl_lab"])
def test_identity_when_ref_equals_target_full_frame(method: str) -> None:
    t = torch.rand(2, 48, 64, 3, dtype=torch.float32)
    r = t.clone()
    out = color_transfer_images(
        t,
        r,
        method=method,
        stats_mode="per_frame",
        target_index=0,
        strength=1.0,
        mask=None,
    )
    torch.testing.assert_close(out, t, rtol=1e-4, atol=1e-4)


def test_histogram_self_similar_when_ref_equals_target() -> None:
    t = torch.rand(2, 48, 64, 3, dtype=torch.float32)
    r = t.clone()
    out = color_transfer_images(
        t,
        r,
        method="histogram",
        stats_mode="per_frame",
        target_index=0,
        strength=1.0,
        mask=None,
    )
    torch.testing.assert_close(out, t, rtol=0.06, atol=0.06)


def test_masked_preserves_pixels_outside_mask_bbox() -> None:
    torch.manual_seed(0)
    t = torch.rand(1, 64, 64, 3, dtype=torch.float32)
    r = torch.rand(1, 64, 64, 3, dtype=torch.float32)
    m = torch.zeros(1, 64, 64, dtype=torch.float32)
    m[:, 24:40, 24:40] = 1.0

    out = color_transfer_images(
        t,
        r,
        method="reinhard_lab",
        stats_mode="per_frame",
        target_index=0,
        strength=1.0,
        mask=m,
        mask_threshold=0.5,
    )

    assert torch.equal(out[0, :24, :, :], t[0, :24, :, :])
    assert torch.equal(out[0, 40:, :, :], t[0, 40:, :, :])
    assert torch.equal(out[0, :, :24, :], t[0, :, :24, :])
    assert torch.equal(out[0, :, 40:, :], t[0, :, 40:, :])


def test_masked_hole_inside_bbox_is_unchanged() -> None:
    torch.manual_seed(2)
    t = torch.rand(1, 64, 64, 3, dtype=torch.float32)
    r = torch.rand(1, 64, 64, 3, dtype=torch.float32)
    m = torch.zeros(1, 64, 64, dtype=torch.float32)
    m[:, 20:44, 20:44] = 1.0
    m[:, 28:36, 28:36] = 0.0

    out = color_transfer_images(
        t,
        r,
        method="reinhard_lab",
        stats_mode="per_frame",
        target_index=0,
        strength=1.0,
        mask=m,
        mask_threshold=0.5,
    )

    assert torch.equal(out[0, 28:36, 28:36, :], t[0, 28:36, 28:36, :])


def test_no_ref_returns_clone() -> None:
    t = torch.ones(1, 8, 8, 3)
    # API expects None ref handled at node level; runtime still receives ref for transfer
    # color_transfer_images early exit needs ref None
    out = color_transfer_images(
        t,
        None,
        method="reinhard_lab",
        stats_mode="per_frame",
        target_index=0,
        strength=1.0,
        mask=None,
    )
    assert torch.equal(out, t)

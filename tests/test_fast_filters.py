"""The GPU blur has to be indistinguishable from scipy, not merely close.

SeamProfileToneMatch feeds this into a tone-matching delta that is added back to the image, so a
systematic edge bias would show up as a seam — which is the exact artefact the node exists to
remove. The border cases get their own assertions because scipy's default ``mode='reflect'`` is
half-sample symmetric and torch's ``F.pad(mode='reflect')`` is not; a port that misses that is
correct in the middle and wrong along every edge.

The fast branch is gated on ``torch.cuda.is_available()``, so on CI these assertions would
otherwise compare scipy against scipy and never touch the code the module exists for.
``forced_fast_path`` opens that branch on whatever device is present.
"""

import contextlib

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from runtime.infer.fast_filters import gaussian, gaussian_stack

TOLERANCE = 2e-4


@contextlib.contextmanager
def forced_fast_path():
    """Run the torch branch on whatever device is present.

    Without this every assertion here compares scipy against scipy: the branch is gated on
    torch.cuda.is_available(), which is False on CI, so the code the commit exists for would
    never be executed by its own tests.
    """
    torch = pytest.importorskip("torch")
    available, to_cuda = torch.cuda.is_available, torch.Tensor.cuda
    torch.cuda.is_available = lambda: True
    if not available():
        torch.Tensor.cuda = lambda self, *args, **kwargs: self
    try:
        yield
    finally:
        torch.cuda.is_available, torch.Tensor.cuda = available, to_cuda


def reference(values, sigma):
    if values.ndim == 2:
        return gaussian_filter(values, sigma)
    return np.stack([gaussian_filter(values[c], sigma) for c in range(values.shape[0])])


@pytest.mark.parametrize("shape", [(64, 64), (3, 64, 64), (3, 97, 61)])
@pytest.mark.parametrize("sigma", [1.0, 4.0, 12.0])
def test_matches_scipy(shape, sigma):
    values = np.random.default_rng(0).random(shape).astype(np.float32)
    with forced_fast_path():
        got = gaussian(values, sigma)
    assert np.abs(reference(values, sigma) - got).max() < TOLERANCE


def test_float64_input_keeps_its_precision():
    """dist * near arrives as float64 from distance_transform_edt, at hundreds of pixels.

    Narrowing to float32 costs ~1e-7 relative, which on that magnitude is ~1e-4 absolute — enough
    to break the equality this module promises.
    """
    dist = np.abs(np.random.default_rng(5).normal(400.0, 120.0, (3, 96, 96)))
    with forced_fast_path():
        got = gaussian(dist, 6.0)
    assert got.dtype == np.float64
    assert np.abs(reference(dist, 6.0) - got).max() < 1e-6


def test_integer_input_goes_to_scipy():
    """Rounding through float and truncating back differs from scipy by 1 LSB on half the pixels."""
    values = np.random.default_rng(6).integers(0, 255, (3, 48, 48), dtype=np.uint8)
    with forced_fast_path():
        got = gaussian(values, 3.0)
    assert np.array_equal(reference(values, 3.0), got)


@pytest.mark.parametrize("sigma", [2.0, 9.0])
def test_edges_match(sigma):
    """A step at the border is where the padding convention shows."""
    values = np.zeros((3, 48, 48), dtype=np.float32)
    values[:, :6, :] = 1.0
    values[:, :, -6:] = 1.0
    with forced_fast_path():
        got = gaussian(values, sigma)
    want = reference(values, sigma)
    for edge in (np.s_[..., :4, :], np.s_[..., -4:, :], np.s_[..., :, :4], np.s_[..., :, -4:]):
        assert np.abs(want[edge] - got[edge]).max() < TOLERANCE


def test_radius_wider_than_image_falls_back():
    """radius = int(4*sigma + 0.5) can exceed the image; the fallback must still be right."""
    values = np.random.default_rng(1).random((3, 12, 12)).astype(np.float32)
    assert np.abs(reference(values, 30.0) - gaussian(values, 30.0)).max() < TOLERANCE


def test_zero_sigma_is_identity():
    values = np.random.default_rng(2).random((3, 16, 16)).astype(np.float32)
    assert np.array_equal(gaussian(values, 0.0), values)


def test_stack_helper_matches_the_loop_it_replaces():
    rng = np.random.default_rng(3)
    values = rng.random((3, 40, 40)).astype(np.float32)
    support = (rng.random((40, 40)) > 0.5).astype(np.float32)
    want = np.stack([gaussian_filter(values[c] * support, 5.0) for c in range(3)])
    with forced_fast_path():
        got = gaussian_stack(values, support, 5.0)
    assert np.abs(want - got).max() < TOLERANCE


def test_dtype_and_shape_preserved():
    values = np.random.default_rng(4).random((3, 32, 32)).astype(np.float32)
    out = gaussian(values, 3.0)
    assert out.shape == values.shape and out.dtype == values.dtype
    flat = gaussian(values[0], 3.0)
    assert flat.shape == values[0].shape

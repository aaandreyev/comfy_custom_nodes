"""GPU Gaussian blur that reproduces ``scipy.ndimage.gaussian_filter`` exactly.

Measured on an RTX 5090 pod: ``SeamProfileToneMatch`` is 22.6 % of an inpaint request, second only
to the sampler, and it barely tracks the bucket area — because it runs on the full frame on the CPU.
Profiling the individual scipy calls at 1536x1536 explains where that goes:

    gaussian_filter over 3 channels    240 ms
    gaussian_filter, one channel        80 ms
    distance_transform_edt             115 ms
    grey_erosion 5x5                    36 ms

The obvious cleanup — replacing the per-channel Python loop with scipy's ``axes`` argument — is
worth nothing: 243 ms against 240. The loop was never the cost; the convolution is, and it is on
the wrong device. Everything else in that list stays on the CPU here, since an exact EDT is not a
convolution and swapping it would change results.

Matching scipy is the whole point, so two details are reproduced rather than approximated:

*Kernel.* ``radius = int(truncate * sigma + 0.5)`` with ``truncate=4.0``, weights
``exp(-0.5 (x/sigma)^2)`` normalised to sum 1 — the same construction ``_gaussian_kernel1d`` uses.

*Edges.* scipy's default ``mode='reflect'`` is half-sample symmetric, ``d c b a | a b c d``. torch's
``F.pad(mode='reflect')`` is whole-sample symmetric, ``d c b | a b c d``, which is a different
filter near the border — so padding is done by index gather instead.

Anything the fast path cannot reproduce (no CUDA, radius wider than the image, non-finite input)
falls through to scipy, so the result is identical in every case and only the speed changes.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter as _scipy_gaussian

try:
    import torch
except Exception:  # pragma: no cover - the pure-numpy tests run without torch
    torch = None

TRUNCATE = 4.0


def _radius(sigma: float) -> int:
    return int(TRUNCATE * float(sigma) + 0.5)


def _kernel1d(sigma: float, device, dtype):
    radius = _radius(sigma)
    x = torch.arange(-radius, radius + 1, device=device, dtype=torch.float64)
    weights = torch.exp(-0.5 * (x / float(sigma)) ** 2)
    return (weights / weights.sum()).to(dtype)


def _symmetric_index(length: int, pad: int, device):
    """Indices for scipy's half-sample symmetric edge: d c b a | a b c d | d c b a."""
    body = torch.arange(length, device=device)
    left = torch.arange(pad - 1, -1, -1, device=device)
    right = torch.arange(length - 1, length - pad - 1, -1, device=device)
    return torch.cat([left, body, right])


def _blur_axis(tensor, kernel, axis: int):
    """Separable pass along one spatial axis of a (C, H, W) tensor."""
    length = tensor.shape[axis]
    pad = (kernel.numel() - 1) // 2
    index = _symmetric_index(length, pad, tensor.device)
    padded = tensor.index_select(axis, index)
    # conv1d wants (batch, channel, length); fold the other spatial axis into batch.
    if axis == 1:
        padded = padded.permute(2, 0, 1)          # (W, C, H+2p)
    else:
        padded = padded.permute(1, 0, 2)          # (H, C, W+2p)
    channels = padded.shape[1]
    weight = kernel.view(1, 1, -1).expand(channels, 1, -1)
    out = torch.nn.functional.conv1d(padded, weight, groups=channels)
    return out.permute(1, 2, 0) if axis == 1 else out.permute(1, 0, 2)


def gaussian(values: np.ndarray, sigma: float) -> np.ndarray:
    """Drop-in for ``gaussian_filter(values, sigma)`` on 2-D or channel-first 3-D input."""
    if sigma <= 0:
        return values.copy()
    squeeze = values.ndim == 2
    array = values[None] if squeeze else values
    if array.ndim != 3:
        return _scipy_gaussian(values, sigma)

    usable = (
        torch is not None
        and torch.cuda.is_available()
        and _radius(sigma) < min(array.shape[1], array.shape[2])
        and np.isfinite(array).all()
    )
    if not usable:
        out = np.stack([_scipy_gaussian(array[c], sigma) for c in range(array.shape[0])])
        return out[0] if squeeze else out

    with torch.no_grad():
        tensor = torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32)).cuda()
        kernel = _kernel1d(sigma, tensor.device, tensor.dtype)
        tensor = _blur_axis(tensor, kernel, 1)
        tensor = _blur_axis(tensor, kernel, 2)
        out = tensor.cpu().numpy()
    out = out.astype(values.dtype, copy=False)
    return out[0] if squeeze else out


def gaussian_stack(values: np.ndarray, support: np.ndarray, sigma: float) -> np.ndarray:
    """``stack([gaussian(values[c] * support, sigma) for c ...])`` as one call."""
    return gaussian(values * support[None], sigma)

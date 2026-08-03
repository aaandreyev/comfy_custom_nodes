"""The composite step must not silently drop generated variants.

edit and outpaint end in ImageCompositeMasked, which sizes its output by the destination. With the
destination coming straight off LoadImage that is batch 1, so three candidates arrive and one
comes out. Reading the count from the latent keeps a single source of truth for it.
"""

import sys
from pathlib import Path

import torch

# nodes/__init__.py pulls in modules that need a live ComfyUI, so the module is imported by path
# the same way the other tests in this suite reach into the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT.parent))

from comfy_custom_nodes_repo.nodes.repeat_image_to_match_batch_node import (
    RepeatImageToMatchBatchNode,
)


def image(batch):
    return torch.arange(batch * 2 * 2 * 3, dtype=torch.float32).reshape(batch, 2, 2, 3)


def latent(batch):
    return {"samples": torch.zeros(batch, 4, 8, 8)}


def test_single_image_repeats_to_candidate_count():
    out, = RepeatImageToMatchBatchNode().apply(image(1), latent(3))
    assert out.shape[0] == 3
    assert torch.equal(out[0], out[1]) and torch.equal(out[1], out[2])


def test_already_matching_batch_is_untouched():
    source = image(3)
    out, = RepeatImageToMatchBatchNode().apply(source, latent(3))
    assert out is source


def test_larger_image_batch_is_left_alone():
    source = image(4)
    out, = RepeatImageToMatchBatchNode().apply(source, latent(2))
    assert out is source


def test_non_divisible_counts_are_trimmed():
    out, = RepeatImageToMatchBatchNode().apply(image(2), latent(5))
    assert out.shape[0] == 5
    assert torch.equal(out[4], image(2)[0])


def test_missing_latent_is_not_fatal():
    source = image(1)
    assert RepeatImageToMatchBatchNode().apply(source, {})[0] is source
    assert RepeatImageToMatchBatchNode().apply(source, None)[0] is source

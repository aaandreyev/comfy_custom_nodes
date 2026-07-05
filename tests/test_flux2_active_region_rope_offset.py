from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT.parent))

from comfy_custom_nodes_repo.nodes.flux2_active_region_rope_offset_node import (
    Flux2ActiveRegionRoPEOffset,
    _token_offset,
    make_shifted_process_img,
)


class _Recorder:
    """Stands in for flux2's process_img; records the offsets it was called with."""

    def __init__(self):
        self.calls = []

    def __call__(self, x, index=0, h_offset=0, w_offset=0, **kwargs):
        self.calls.append({"x": x, "index": index, "h_offset": h_offset, "w_offset": w_offset, **kwargs})
        return "img", "ids"


class _FakeNativeDiffusionModel:
    def __init__(self):
        self.process_img = _Recorder()


class _FakeNunchakuWrapper:
    ctx_for_copy = {}


class _FakeInnerModel:
    def __init__(self, diffusion_model):
        self.diffusion_model = diffusion_model


class _FakeModelPatcher:
    def __init__(self, diffusion_model):
        self.model = _FakeInnerModel(diffusion_model)
        self.model_options = {}
        self.object_patches = {}

    def clone(self):
        c = _FakeModelPatcher(self.model.diffusion_model)
        c.model = self.model
        c.model_options = {k: dict(v) if isinstance(v, dict) else v for k, v in self.model_options.items()}
        return c

    def add_object_patch(self, name, obj):
        self.object_patches[name] = obj


def test_token_offset_converts_pixels_to_tokens():
    assert _token_offset(0) == 0
    assert _token_offset(16) == 1
    assert _token_offset(1024) == 64


def test_token_offset_floors_non_multiples():
    assert _token_offset(17) == 1
    assert _token_offset(15) == 0


def test_shifted_process_img_shifts_only_main_image():
    rec = _Recorder()
    patched = make_shifted_process_img(rec, dy_tokens=3, dx_tokens=5)

    patched("main")
    patched("ref", index=10.0)
    patched("main2", index=0, h_offset=2, w_offset=1)

    main, ref, main2 = rec.calls
    assert (main["h_offset"], main["w_offset"]) == (3, 5)
    assert (ref["h_offset"], ref["w_offset"]) == (0, 0) and ref["index"] == 10.0
    assert (main2["h_offset"], main2["w_offset"]) == (5, 6)


def test_shifted_process_img_forwards_extra_kwargs():
    rec = _Recorder()
    patched = make_shifted_process_img(rec, dy_tokens=1, dx_tokens=1)
    patched("main", transformer_options={"a": 1})
    assert rec.calls[0]["transformer_options"] == {"a": 1}


def test_apply_zero_offset_returns_model_unchanged():
    model = _FakeModelPatcher(_FakeNativeDiffusionModel())
    (out,) = Flux2ActiveRegionRoPEOffset().apply(model, x_offset_px=0, y_offset_px=0)
    assert out is model
    assert model.object_patches == {}


def test_apply_native_model_installs_object_patch_and_options():
    dm = _FakeNativeDiffusionModel()
    model = _FakeModelPatcher(dm)
    (out,) = Flux2ActiveRegionRoPEOffset().apply(model, x_offset_px=1024, y_offset_px=512)

    assert out is not model
    assert out.model_options["transformer_options"]["flux2_active_offset"] == (32, 64)
    patched = out.object_patches["diffusion_model.process_img"]
    patched("main")
    assert (dm.process_img.calls[0]["h_offset"], dm.process_img.calls[0]["w_offset"]) == (32, 64)


def test_apply_nunchaku_wrapper_uses_transformer_options_only():
    model = _FakeModelPatcher(_FakeNunchakuWrapper())
    (out,) = Flux2ActiveRegionRoPEOffset().apply(model, x_offset_px=256, y_offset_px=256)
    assert out.model_options["transformer_options"]["flux2_active_offset"] == (16, 16)
    assert out.object_patches == {}


def test_apply_rejects_non_flux2_model():
    class _Alien:
        pass

    model = _FakeModelPatcher(_Alien())
    with pytest.raises(ValueError, match="process_img"):
        Flux2ActiveRegionRoPEOffset().apply(model, x_offset_px=16, y_offset_px=0)

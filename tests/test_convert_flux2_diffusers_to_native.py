"""Round-trip proof for the FLUX.2 diffusers->native converter.

Forward direction (native -> diffusers) is a faithful vendored copy of diffusers'
``convert_flux2_transformer_checkpoint_to_diffusers`` (loaders/single_file_utils.py). The tool
under test must invert it exactly: native -> forward -> inverse == identity.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT.parent))

from comfy_custom_nodes_repo.tools.convert_flux2_diffusers_to_native import (
    convert_state_dict,
    is_diffusers_layout,
    is_native_layout,
)

D = 8  # tiny hidden size for the fake checkpoint


# --- vendored official forward conversion (native -> diffusers) -----------------------------

_RENAME = {
    "img_in": "x_embedder",
    "txt_in": "context_embedder",
    "time_in.in_layer": "time_guidance_embed.timestep_embedder.linear_1",
    "time_in.out_layer": "time_guidance_embed.timestep_embedder.linear_2",
    "guidance_in.in_layer": "time_guidance_embed.guidance_embedder.linear_1",
    "guidance_in.out_layer": "time_guidance_embed.guidance_embedder.linear_2",
    "double_stream_modulation_img.lin": "double_stream_modulation_img.linear",
    "double_stream_modulation_txt.lin": "double_stream_modulation_txt.linear",
    "single_stream_modulation.lin": "single_stream_modulation.linear",
    "final_layer.linear": "proj_out",
}

_DOUBLE = {
    "img_attn.norm.query_norm": "attn.norm_q",
    "img_attn.norm.key_norm": "attn.norm_k",
    "img_attn.proj": "attn.to_out.0",
    "img_mlp.0": "ff.linear_in",
    "img_mlp.2": "ff.linear_out",
    "txt_attn.norm.query_norm": "attn.norm_added_q",
    "txt_attn.norm.key_norm": "attn.norm_added_k",
    "txt_attn.proj": "attn.to_add_out",
    "txt_mlp.0": "ff_context.linear_in",
    "txt_mlp.2": "ff_context.linear_out",
}

_SINGLE = {
    "linear1": "attn.to_qkv_mlp_proj",
    "linear2": "attn.to_out",
    "norm.query_norm": "attn.norm_q",
    "norm.key_norm": "attn.norm_k",
}


def _swap(weight):
    shift, scale = weight.chunk(2, dim=0)
    return torch.cat([scale, shift], dim=0)


def official_native_to_diffusers(native: dict) -> dict:
    sd = dict(native)
    out = {}
    for key in list(sd.keys()):
        new_key = key
        for old, new in _RENAME.items():
            new_key = new_key.replace(old, new)
        out[new_key] = sd.pop(key)

    for key in list(out.keys()):
        if "adaLN_modulation" in key:
            out["norm_out.linear." + key.rsplit(".", 1)[1]] = _swap(out.pop(key))
        elif key.startswith("double_blocks."):
            parts = key.split(".")
            idx, within, param = parts[1], ".".join(parts[2:-1]), parts[-1]
            if param == "scale":
                param = "weight"
            val = out.pop(key)
            if "qkv" in within:
                q, k, v = torch.chunk(val, 3, dim=0)
                names = (
                    ("attn.to_q", "attn.to_k", "attn.to_v")
                    if "img" in within
                    else ("attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj")
                )
                for name, t in zip(names, (q, k, v)):
                    out[f"transformer_blocks.{idx}.{name}.{param}"] = t
            else:
                out[f"transformer_blocks.{idx}.{_DOUBLE[within]}.{param}"] = val
        elif key.startswith("single_blocks."):
            parts = key.split(".")
            idx, within, param = parts[1], ".".join(parts[2:-1]), parts[-1]
            if param == "scale":
                param = "weight"
            out[f"single_transformer_blocks.{idx}.{_SINGLE[within]}.{param}"] = out.pop(key)
    return out


# --- fake native checkpoint ------------------------------------------------------------------


def fake_native_sd() -> dict:
    g = torch.Generator().manual_seed(7)

    def t(*shape):
        return torch.randn(*shape, generator=g)

    sd = {
        "img_in.weight": t(D, 16),
        "txt_in.weight": t(D, 24),
        "time_in.in_layer.weight": t(D, 4),
        "time_in.out_layer.weight": t(D, D),
        "guidance_in.in_layer.weight": t(D, 4),
        "guidance_in.out_layer.weight": t(D, D),
        "double_stream_modulation_img.lin.weight": t(6 * D, D),
        "double_stream_modulation_txt.lin.weight": t(6 * D, D),
        "single_stream_modulation.lin.weight": t(3 * D, D),
        "final_layer.linear.weight": t(16, D),
        "final_layer.adaLN_modulation.1.weight": t(2 * D, D),
    }
    for n in range(2):
        sd.update({
            f"double_blocks.{n}.img_attn.qkv.weight": t(3 * D, D),
            f"double_blocks.{n}.img_attn.norm.query_norm.scale": t(D),
            f"double_blocks.{n}.img_attn.norm.key_norm.scale": t(D),
            f"double_blocks.{n}.img_attn.proj.weight": t(D, D),
            f"double_blocks.{n}.img_mlp.0.weight": t(3 * D, D),
            f"double_blocks.{n}.img_mlp.2.weight": t(D, 3 * D),
            f"double_blocks.{n}.txt_attn.qkv.weight": t(3 * D, D),
            f"double_blocks.{n}.txt_attn.norm.query_norm.scale": t(D),
            f"double_blocks.{n}.txt_attn.norm.key_norm.scale": t(D),
            f"double_blocks.{n}.txt_attn.proj.weight": t(D, D),
            f"double_blocks.{n}.txt_mlp.0.weight": t(3 * D, D),
            f"double_blocks.{n}.txt_mlp.2.weight": t(D, 3 * D),
        })
    for n in range(3):
        sd.update({
            f"single_blocks.{n}.linear1.weight": t(6 * D, D),
            f"single_blocks.{n}.linear2.weight": t(D, 4 * D),
            f"single_blocks.{n}.norm.query_norm.scale": t(D),
            f"single_blocks.{n}.norm.key_norm.scale": t(D),
        })
    return sd


# --- tests ------------------------------------------------------------------------------------


def test_layout_detection():
    native = fake_native_sd()
    diff = official_native_to_diffusers(fake_native_sd())
    assert is_native_layout(native) and not is_diffusers_layout(native)
    assert is_diffusers_layout(diff) and not is_native_layout(diff)


def test_forward_produces_expected_diffusers_markers():
    diff = official_native_to_diffusers(fake_native_sd())
    assert "x_embedder.weight" in diff
    assert "x_embedder.bias" not in diff  # the exact key ComfyUI's converter crashes on
    assert "transformer_blocks.0.attn.to_q.weight" in diff
    assert "single_transformer_blocks.0.attn.to_qkv_mlp_proj.weight" in diff
    assert "norm_out.linear.weight" in diff


def test_round_trip_is_identity():
    native = fake_native_sd()
    recovered = convert_state_dict(official_native_to_diffusers(fake_native_sd()))
    assert set(recovered) == set(native), (
        f"missing={sorted(set(native) - set(recovered))[:5]} "
        f"extra={sorted(set(recovered) - set(native))[:5]}"
    )
    for k in native:
        assert torch.equal(recovered[k], native[k]), f"tensor mismatch at {k}"


def test_unknown_block_key_raises():
    diff = official_native_to_diffusers(fake_native_sd())
    diff["transformer_blocks.0.attn.some_new_thing.weight"] = torch.zeros(1)
    try:
        convert_state_dict(diff)
    except KeyError as e:
        assert "some_new_thing" in str(e)
    else:
        raise AssertionError("expected KeyError for unmapped block key")


def test_svdquant_checkpoint_is_detected():
    from comfy_custom_nodes_repo.tools.convert_flux2_diffusers_to_native import is_svdquant_checkpoint

    svdq_keys = [
        "transformer_blocks.0.attn.to_qkv.qweight",
        "transformer_blocks.0.attn.to_qkv.wscales",
        "transformer_blocks.0.attn.to_qkv.proj_down",
        "single_transformer_blocks.0.qkv_proj.smooth",
    ]
    assert is_svdquant_checkpoint(svdq_keys)
    assert not is_svdquant_checkpoint(list(fake_native_sd()))
    assert not is_svdquant_checkpoint(list(official_native_to_diffusers(fake_native_sd())))

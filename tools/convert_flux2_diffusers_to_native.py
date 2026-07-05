#!/usr/bin/env python3
"""Convert a FLUX.2 transformer checkpoint from diffusers layout to BFL-native (ComfyUI) layout.

Why: BFL published the FLUX.2-klein-9b-kv checkpoints (bf16 root file AND the -fp8 repack) in
*diffusers* layout (``x_embedder.*``, ``transformer_blocks.*``). ComfyUI's UNETLoader only detects
the BFL-native layout (``img_in.*``, ``double_blocks.*``) and its generic diffusers converter has
no flux2 branch — it crashes with ``KeyError: 'x_embedder.bias'`` (flux2 has no biases). This tool
is the exact inverse of diffusers' official
``convert_flux2_transformer_checkpoint_to_diffusers`` (loaders/single_file_utils.py), so a
round-trip native -> diffusers -> native is identity (covered by the unit test).

Usage (in-place, idempotent — native input passes through untouched):
    python3 convert_flux2_diffusers_to_native.py model.safetensors [output.safetensors]

Memory: holds one full copy of the state dict (~18 GB for klein-9B bf16) — fine on L4/A100 Colab
RAM, do not run on a base 12 GB runtime.
"""

from __future__ import annotations

import sys

import torch

# Inverse of FLUX2_TRANSFORMER_KEYS_RENAME_DICT (diffusers -> native). Order matters: longest
# prefixes first so e.g. "time_guidance_embed.timestep_embedder.linear_1" wins over any shorter
# accidental overlap.
RENAME_DIFFUSERS_TO_NATIVE = [
    ("time_guidance_embed.timestep_embedder.linear_1", "time_in.in_layer"),
    ("time_guidance_embed.timestep_embedder.linear_2", "time_in.out_layer"),
    ("time_guidance_embed.guidance_embedder.linear_1", "guidance_in.in_layer"),
    ("time_guidance_embed.guidance_embedder.linear_2", "guidance_in.out_layer"),
    ("double_stream_modulation_img.linear", "double_stream_modulation_img.lin"),
    ("double_stream_modulation_txt.linear", "double_stream_modulation_txt.lin"),
    ("single_stream_modulation.linear", "single_stream_modulation.lin"),
    ("x_embedder", "img_in"),
    ("context_embedder", "txt_in"),
    ("proj_out", "final_layer.linear"),
]

# transformer_blocks.{N}.<diffusers> -> double_blocks.{N}.<native>; norms become ".scale" params.
DOUBLE_BLOCK_MAP = {
    "attn.norm_q": ("img_attn.norm.query_norm", True),
    "attn.norm_k": ("img_attn.norm.key_norm", True),
    "attn.to_out.0": ("img_attn.proj", False),
    "ff.linear_in": ("img_mlp.0", False),
    "ff.linear_out": ("img_mlp.2", False),
    "attn.norm_added_q": ("txt_attn.norm.query_norm", True),
    "attn.norm_added_k": ("txt_attn.norm.key_norm", True),
    "attn.to_add_out": ("txt_attn.proj", False),
    "ff_context.linear_in": ("txt_mlp.0", False),
    "ff_context.linear_out": ("txt_mlp.2", False),
}

SINGLE_BLOCK_MAP = {
    "attn.to_qkv_mlp_proj": ("linear1", False),
    "attn.to_out": ("linear2", False),
    "attn.norm_q": ("norm.query_norm", True),
    "attn.norm_k": ("norm.key_norm", True),
}

QKV_GROUPS = {
    # native fused name -> (diffusers q, k, v) within a double block
    "img_attn.qkv": ("attn.to_q", "attn.to_k", "attn.to_v"),
    "txt_attn.qkv": ("attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj"),
}


def is_native_layout(keys) -> bool:
    return any(k.startswith("img_in.") or k.startswith("double_blocks.") for k in keys)


def is_diffusers_layout(keys) -> bool:
    return any(k.startswith("x_embedder.") or k.startswith("transformer_blocks.") for k in keys)


def _swap_scale_shift(weight: torch.Tensor) -> torch.Tensor:
    # diffusers stores (scale, shift); native stores (shift, scale). Self-inverse.
    scale, shift = weight.chunk(2, dim=0)
    return torch.cat([shift, scale], dim=0)


def convert_state_dict(sd: dict) -> dict:
    """diffusers-layout FLUX.2 state dict -> BFL-native. Unknown keys pass through unchanged."""
    out: dict = {}
    leftovers: dict[str, dict[str, torch.Tensor]] = {}

    for key in list(sd.keys()):
        val = sd.pop(key)

        if key.startswith("norm_out.linear."):
            # native final_layer.adaLN_modulation.1 with (shift, scale) order; bias swaps too
            param = key.rsplit(".", 1)[1]
            out[f"final_layer.adaLN_modulation.1.{param}"] = _swap_scale_shift(val)
            continue

        if key.startswith("transformer_blocks."):
            _, idx, *rest = key.split(".")
            within = ".".join(rest[:-1])
            param = rest[-1]
            hit = DOUBLE_BLOCK_MAP.get(within)
            if hit is not None:
                native_within, is_norm = hit
                native_param = "scale" if (is_norm and param == "weight") else param
                out[f"double_blocks.{idx}.{native_within}.{native_param}"] = val
                continue
            leftovers.setdefault(f"double_blocks.{idx}", {})[f"{within}.{param}"] = val
            continue

        if key.startswith("single_transformer_blocks."):
            _, idx, *rest = key.split(".")
            within = ".".join(rest[:-1])
            param = rest[-1]
            hit = SINGLE_BLOCK_MAP.get(within)
            if hit is not None:
                native_within, is_norm = hit
                native_param = "scale" if (is_norm and param == "weight") else param
                out[f"single_blocks.{idx}.{native_within}.{native_param}"] = val
                continue
            leftovers.setdefault(f"single_blocks.{idx}", {})[f"{within}.{param}"] = val
            continue

        new_key = key
        for diff_name, native_name in RENAME_DIFFUSERS_TO_NATIVE:
            if diff_name in new_key:
                new_key = new_key.replace(diff_name, native_name)
                break
        out[new_key] = val

    # Re-fuse the QKV projections that diffusers chunked apart.
    for block_prefix, parts in leftovers.items():
        consumed = set()
        for native_fused, (qn, kn, vn) in QKV_GROUPS.items():
            for param in ("weight", "bias"):
                names = (f"{qn}.{param}", f"{kn}.{param}", f"{vn}.{param}")
                if all(n in parts for n in names):
                    out[f"{block_prefix}.{native_fused}.{param}"] = torch.cat(
                        [parts[n] for n in names], dim=0
                    )
                    consumed.update(names)
        unknown = set(parts) - consumed
        if unknown:
            raise KeyError(f"unmapped keys in {block_prefix}: {sorted(unknown)}")

    return out


def convert_file(src: str, dst: str | None = None) -> str:
    import os

    from safetensors import safe_open
    from safetensors.torch import save_file

    with safe_open(src, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        if is_native_layout(keys):
            print(f"{src}: already BFL-native layout, nothing to do")
            return src
        if not is_diffusers_layout(keys):
            raise ValueError(f"{src}: neither native nor diffusers FLUX.2 layout")
        sd = {k: f.get_tensor(k) for k in keys}
        metadata = f.metadata()

    out = convert_state_dict(sd)
    print(f"{src}: converted {len(keys)} diffusers keys -> {len(out)} native keys")

    target = dst or src
    tmp = target + ".converting"
    save_file(out, tmp, metadata=dict(metadata) if metadata else None)
    os.replace(tmp, target)
    print(f"wrote {target}")
    return target


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    convert_file(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)

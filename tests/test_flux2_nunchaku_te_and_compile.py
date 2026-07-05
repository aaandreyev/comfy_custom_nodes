from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT.parent))

from comfy_custom_nodes_repo.nodes.flux2_nunchaku_te_loader_node import (
    NunchakuKleinTEShim,
    assemble_klein_conditioning,
    tokens_to_tensor,
)
from comfy_custom_nodes_repo.nodes.flux2_compile_nodes import (
    Flux2CLIPCompile,
    NunchakuFlux2ModelCompile,
    compile_blocks,
    resolve_clip_inner,
)

B, T, D = 1, 16, 8
N_LAYERS = 36
PAD = 151643


class _FakeQwen(nn.Module):
    """Returns deterministic hidden_states: layer k output filled with value k (embeddings=0)."""

    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))
        self.calls = []

    def forward(self, input_ids=None, attention_mask=None, output_hidden_states=None, return_dict=None):
        self.calls.append({"ids": input_ids, "mask": attention_mask, "ohs": output_hidden_states})
        hs = tuple(
            torch.full((input_ids.shape[0], input_ids.shape[1], D), float(k))
            for k in range(N_LAYERS + 1)
        )
        return SimpleNamespace(hidden_states=hs)


def test_layer_index_mapping_comfy_to_transformers():
    hs = tuple(torch.full((B, T, D), float(k)) for k in range(N_LAYERS + 1))
    out = assemble_klein_conditioning(hs, layers=(9, 18, 27))
    # comfy layer i == transformers hidden_states[i+1]
    assert out.shape == (B, T, 3 * D)
    assert torch.all(out[..., 0:D] == 10.0)
    assert torch.all(out[..., D:2 * D] == 19.0)
    assert torch.all(out[..., 2 * D:] == 28.0)


def test_assembly_matches_comfy_flux2_te_math():
    g = torch.Generator().manual_seed(3)
    hs = tuple(torch.randn(B, T, D, generator=g) for _ in range(N_LAYERS + 1))
    out = assemble_klein_conditioning(hs, layers=(9, 18, 27))
    # reference: comfy Flux2TEModel.encode_token_weights math on the same three states
    ref = torch.stack((hs[10], hs[19], hs[28]), dim=1).movedim(1, 2)
    ref = ref.reshape(ref.shape[0], ref.shape[1], -1)
    torch.testing.assert_close(out, ref)


def test_tokens_to_tensor_handles_comfy_dict_and_bare_lists():
    pairs = [[(5, 1.0), (6, 1.0), (PAD, 1.0)]]
    assert tokens_to_tensor({"qwen3_8b": pairs}).tolist() == [[5, 6, PAD]]
    assert tokens_to_tensor(pairs).tolist() == [[5, 6, PAD]]


def test_shim_encode_token_weights_end_to_end():
    inner = _FakeQwen()
    shim = NunchakuKleinTEShim(inner, layers=(9, 18, 27))
    cond, pooled, extra = shim.encode_token_weights({"qwen3_8b": [[(5, 1.0), (7, 1.0), (PAD, 1.0)]]})
    assert cond.shape == (1, 3, 3 * D) and cond.dtype == torch.float32
    assert pooled is None
    assert extra["attention_mask"].tolist() == [[1, 1, 0]]  # pad masked out
    call = inner.calls[0]
    assert call["ohs"] is True and call["ids"].tolist() == [[5, 7, PAD]]


def _marking_compile(fn, backend, mode, dynamic):
    def wrapped(*a, **k):
        return fn(*a, **k)

    wrapped._is_compiled_fn = True
    wrapped._backend = backend
    return wrapped


def test_compile_blocks_wraps_forwards_keeps_module_tree_and_is_idempotent():
    blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])
    names_before = [n for n, _ in blocks.named_modules()]
    n = compile_blocks(blocks, "inductor", "default", True, compile_fn=_marking_compile)
    assert n == 3 and all(getattr(b, "_flux2_compiled", False) for b in blocks)
    assert all(isinstance(b, nn.Linear) for b in blocks)  # modules NOT replaced
    assert [n for n, _ in blocks.named_modules()] == names_before  # name-keyed LoRA machinery safe
    assert all(getattr(b.forward, "_is_compiled_fn", False) for b in blocks)
    wrapped_fwd = blocks[0].forward
    n2 = compile_blocks(blocks, "inductor", "default", True, compile_fn=_marking_compile)
    assert n2 == 3 and blocks[0].forward is wrapped_fwd  # not double-wrapped
    x = torch.randn(2, 4)
    torch.testing.assert_close(blocks[0](x), nn.functional.linear(x, blocks[0].weight, blocks[0].bias))


def test_compile_blocks_survives_a_failing_block():
    def flaky(fn, backend, mode, dynamic):
        if getattr(fn.__self__, "in_features", 0) == 99:
            raise RuntimeError("dynamo says no")
        return _marking_compile(fn, backend, mode, dynamic)

    blocks = nn.ModuleList([nn.Linear(4, 4), nn.Linear(99, 4), nn.Linear(4, 4)])
    n = compile_blocks(blocks, "inductor", "default", True, compile_fn=flaky)
    assert n == 2
    assert not getattr(blocks[1], "_flux2_compiled", False)  # failed block left eager


def _fake_nunchaku_model():
    transformer = SimpleNamespace(
        transformer_blocks=nn.ModuleList([nn.Linear(4, 4) for _ in range(2)]),
        single_transformer_blocks=nn.ModuleList([nn.Linear(4, 4) for _ in range(3)]),
    )
    wrapper = SimpleNamespace(model=transformer, ctx_for_copy={})
    return SimpleNamespace(model=SimpleNamespace(diffusion_model=wrapper))


def test_model_compile_node_wraps_both_block_lists():
    model = _fake_nunchaku_model()
    (out,) = NunchakuFlux2ModelCompile().apply(model, True, "inductor", "default", True, compile_fn=_marking_compile)
    tr = out.model.diffusion_model.model
    assert all(getattr(b, "_flux2_compiled", False) for b in tr.transformer_blocks)
    assert all(getattr(b, "_flux2_compiled", False) for b in tr.single_transformer_blocks)


def test_model_compile_node_disabled_is_passthrough():
    model = _fake_nunchaku_model()
    (out,) = NunchakuFlux2ModelCompile().apply(model, False, "inductor", "default", True, compile_fn=_marking_compile)
    assert not getattr(out.model.diffusion_model.model.transformer_blocks[0], "_flux2_compiled", False)


def test_model_compile_node_rejects_non_nunchaku():
    model = SimpleNamespace(model=SimpleNamespace(diffusion_model=SimpleNamespace(model=None)))
    with pytest.raises(ValueError, match="Nunchaku"):
        NunchakuFlux2ModelCompile().apply(model, True, "inductor", "default", True)


def test_clip_compile_resolves_fp8_and_shim_paths():
    # SD1ClipModel records its submodule attr in .clip — the resolver must use it
    fp8_csm = SimpleNamespace(clip="qwen3_8b", qwen3_8b=SimpleNamespace(transformer=nn.Linear(2, 2)))
    path, inner = resolve_clip_inner(fp8_csm)
    assert path == "qwen3_8b.transformer" and isinstance(inner, nn.Linear)

    shim_csm = NunchakuKleinTEShim(_FakeQwen())
    path, inner = resolve_clip_inner(shim_csm)
    assert path == "inner" and isinstance(inner, _FakeQwen)


def test_clip_compile_resolves_via_named_children_fallback():
    class CSM(nn.Module):
        def __init__(self):
            super().__init__()
            self.renamed_te = nn.Module()
            self.renamed_te.transformer = nn.Linear(2, 2)

    path, inner = resolve_clip_inner(CSM())
    assert path == "renamed_te.transformer" and isinstance(inner, nn.Linear)


def test_clip_compile_unresolvable_error_names_children():
    class Alien(nn.Module):
        def __init__(self):
            super().__init__()
            self.something_768 = nn.Linear(2, 2)

    clip = SimpleNamespace(cond_stage_model=Alien())
    with pytest.raises(ValueError, match="something_768"):
        Flux2CLIPCompile().apply(clip, True, "inductor", "default", True, compile_fn=_marking_compile)


def test_clip_compile_node_wraps_inner_forward_in_place():
    lin = nn.Linear(2, 2)
    clip = SimpleNamespace(cond_stage_model=SimpleNamespace(clip="qwen3_8b", qwen3_8b=SimpleNamespace(transformer=lin)))
    (out,) = Flux2CLIPCompile().apply(clip, True, "inductor", "default", True, compile_fn=_marking_compile)
    inner = out.cond_stage_model.qwen3_8b.transformer
    assert inner is lin  # module identity preserved (ModelPatcher paths intact)
    assert getattr(inner, "_flux2_compiled", False) and getattr(inner.forward, "_is_compiled_fn", False)
    wrapped_fwd = inner.forward
    (out2,) = Flux2CLIPCompile().apply(out, True, "inductor", "default", True, compile_fn=_marking_compile)
    assert out2.cond_stage_model.qwen3_8b.transformer.forward is wrapped_fwd

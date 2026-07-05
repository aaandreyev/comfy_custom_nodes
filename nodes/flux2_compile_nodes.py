"""
torch.compile nodes for the FLUX.2 nunchaku path.

Honest expectations, per component:

* **CLIP compile** (:class:`Flux2CLIPCompile`) — the fp8 Qwen3-8B TE is a plain torch model, so
  torch.compile applies fully. Encode runs once per prompt; the win is real but small in absolute
  terms (hundreds of ms). First encode after (re)start pays a compile warmup.
* **DiT compile** (:class:`NunchakuFlux2ModelCompile`) — the nunchaku transformer's linears are
  custom CUDA kernels that dynamo cannot trace, so whole-model compile would be all graph breaks.
  Instead this node compiles each transformer block individually (norms, rope math, elementwise
  glue fuse; the quant matmuls stay opaque). Expected gain is modest (single-digit %) and can be
  negative on some stacks — A/B it. ``enabled=False`` passes through untouched.
* **VAE compile** — use KJNodes' ``TorchCompileVAE`` (already installed); no node here.

Both nodes mutate the loaded module in place (the nunchaku transformer is a shared, loader-cached
object) — that is deliberate: one resident model, compiled once, reused across prompts. Restart
ComfyUI to undo.

GPU-VERIFY: on the pod, generate with enabled=True vs False on the same seed — images must match
(compile is numerically near-exact); timing per step logged by ComfyUI shows the delta. If dynamo
crashes inside a block, the node logs the block index and leaves that block eager.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def _default_compile(module, backend: str, mode: str, dynamic: bool):
    kwargs = {"backend": backend, "dynamic": dynamic}
    if backend == "inductor" and mode != "default":
        kwargs["mode"] = mode
    return torch.compile(module, **kwargs)


def compile_blocks(block_list, backend: str, mode: str, dynamic: bool, compile_fn=_default_compile) -> int:
    """Compile each block's ``forward`` in place; any block that fails stays eager.

    Deliberately wraps the *bound forward*, not the module: replacing modules with
    OptimizedModule renames the tree (``_orig_mod.`` prefixes), which breaks every name-keyed
    consumer — nunchaku's LoRA param routing and the fork wrapper's ``_restore_lora_base``
    snapshot. Forward-wrapping keeps the module tree byte-identical; LoRA rank concats change
    tensor shapes, which ``dynamic=True`` absorbs via recompile instead of wrong output.
    """
    compiled = 0
    for i in range(len(block_list)):
        block = block_list[i]
        if getattr(block, "_flux2_compiled", False):
            compiled += 1
            continue
        try:
            block.forward = compile_fn(block.forward, backend, mode, dynamic)
            block._flux2_compiled = True
            compiled += 1
        except Exception as exc:
            logger.warning("block %d left eager (compile failed: %s)", i, exc)
    return compiled


class NunchakuFlux2ModelCompile:
    """Per-block torch.compile for a Nunchaku FLUX.2 model (quant kernels stay opaque)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Model from `Nunchaku FLUX.2 DiT Loader`."}),
                "enabled": ("BOOLEAN", {"default": True}),
                "backend": (["inductor", "cudagraphs"], {"default": "inductor"}),
                "mode": (["default", "max-autotune", "reduce-overhead"], {"default": "default"}),
                "dynamic": ("BOOLEAN", {"default": True, "tooltip": "Survive resolution/KV-mode changes without recompiles."}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "flux2/nunchaku"
    TITLE = "Nunchaku FLUX.2 Model Compile (per-block)"

    def apply(self, model, enabled: bool, backend: str, mode: str, dynamic: bool, compile_fn=_default_compile):
        if not enabled:
            return (model,)
        wrapper = model.model.diffusion_model
        transformer = getattr(wrapper, "model", None)
        if transformer is None or not hasattr(transformer, "transformer_blocks"):
            raise ValueError("Not a Nunchaku FLUX.2 model — feed the output of `Nunchaku FLUX.2 DiT Loader`.")
        n_double = compile_blocks(transformer.transformer_blocks, backend, mode, dynamic, compile_fn)
        n_single = compile_blocks(transformer.single_transformer_blocks, backend, mode, dynamic, compile_fn)
        logger.info("nunchaku DiT compile: %d double + %d single blocks wrapped (backend=%s)", n_double, n_single, backend)
        return (model,)


_CLIP_INNER_PATHS = (
    "qwen3_8b.transformer",   # comfy fp8 klein TE (Qwen3-8B)
    "qwen3_4b.transformer",   # klein 4B TE
    "mistral3_24b.transformer",  # flux2 dev TE
    "inner",                  # NunchakuKleinTEShim (int4 TE)
)


def resolve_clip_inner(cond_stage_model):
    """Find the heavy LLM submodule inside a comfy cond_stage_model. Returns (path, module)."""
    for path in _CLIP_INNER_PATHS:
        obj = cond_stage_model
        ok = True
        for part in path.split("."):
            if not hasattr(obj, part):
                ok = False
                break
            obj = getattr(obj, part)
        if ok:
            return path, obj
    return None, None


class Flux2CLIPCompile:
    """torch.compile the text encoder's LLM (Qwen3/Mistral) inside a comfy CLIP."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "enabled": ("BOOLEAN", {"default": True}),
                "backend": (["inductor", "cudagraphs"], {"default": "inductor"}),
                "mode": (["default", "max-autotune", "reduce-overhead"], {"default": "default"}),
                "dynamic": ("BOOLEAN", {"default": True, "tooltip": "Prompt length varies — keep True."}),
            }
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "apply"
    CATEGORY = "flux2/nunchaku"
    TITLE = "Flux2 CLIP Compile"

    def apply(self, clip, enabled: bool, backend: str, mode: str, dynamic: bool, compile_fn=_default_compile):
        if not enabled:
            return (clip,)
        path, inner = resolve_clip_inner(clip.cond_stage_model)
        if inner is None:
            raise ValueError("No known LLM submodule found in this CLIP (expected a FLUX.2 klein/dev text encoder).")
        if getattr(inner, "_flux2_compiled", False):
            return (clip,)
        # Wrap the bound forward, not the module — keeps the module tree (and every name-keyed
        # weight/patch path in comfy's CLIP ModelPatcher) untouched.
        inner.forward = compile_fn(inner.forward, backend, mode, dynamic)
        inner._flux2_compiled = True
        logger.info("CLIP compile: wrapped %s.forward (backend=%s)", path, backend)
        return (clip,)

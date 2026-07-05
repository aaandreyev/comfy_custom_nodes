"""
:class:`NunchakuQwen3TELoader` — load the AWQ-int4 Qwen3 text encoder
(``svdq-int4-Qwen3-text-Nunchaku.safetensors``, tonera/Qwen3-text-Nunchaku) as a ComfyUI CLIP for
FLUX.2 [klein] 9B.

Why a dedicated node: the file packs int4 weights (qweight/scales/scaled_zeros, AWQ tinychat
W4A16) that only the nunchaku runtime can execute — the standard CLIPLoader builds a plain torch
model and produces garbage (the infamous 512x768 conditioning). This node loads the checkpoint
through vitoom-nunchaku's own ``NunchakuQwenEncoderModel`` runtime and adapts it to the comfy CLIP
interface, reusing comfy's KleinTokenizer8B so the chat template and padding are identical to the
fp8 path.

Layer-index parity: comfy's klein TE takes hidden states at ``layer=[9, 18, 27]`` where index i
means "after transformer block i". transformers' ``output_hidden_states`` indexes from the
embeddings (hidden_states[0] = embeddings, hidden_states[k] = output of block k-1), so comfy layer
i == hidden_states[i + 1]. The final 12288-dim conditioning is stack([h9, h18, h27], dim=1)
.movedim(1, 2).reshape(B, T, -1) — byte-for-byte the math in comfy's Flux2TEModel.

GPU-VERIFY (decisive parity gate — run on the pod before trusting this in production):
  1. Encode the same prompt via CLIPLoader(qwen_3_8b_fp8mixed, type=flux2) and via this node;
     cosine similarity of the two [1, 512, 12288] conditionings must be > 0.99
     (int4 quantization noise accounts for the rest).
  2. A full 4-step generate with this TE must be visually equivalent to the fp8-TE run
     (same seed, LPIPS < 0.1).
  3. VRAM: expect ~3 GiB resident instead of ~8.9 GiB for the fp8 TE.
"""

from __future__ import annotations

import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)

DEFAULT_LAYERS = (9, 18, 27)  # comfy indexing: hidden state after block i
PAD_TOKEN = 151643


def assemble_klein_conditioning(hidden_states, layers=DEFAULT_LAYERS):
    """[B,T,4096] x3 (comfy layers i -> transformers hidden_states[i+1]) -> [B, T, 12288].

    ``hidden_states`` is the transformers-style tuple (embeddings first).
    """
    picked = [hidden_states[i + 1] for i in layers]
    out = torch.stack(picked, dim=1)          # [B, 3, T, D]
    out = out.movedim(1, 2)                   # [B, T, 3, D]
    return out.reshape(out.shape[0], out.shape[1], -1)  # [B, T, 3D]


def tokens_to_tensor(token_weight_pairs, embedding_key: str = "qwen3_8b"):
    """comfy token_weight_pairs (dict keyed by encoder name, or bare list) -> LongTensor ids."""
    if isinstance(token_weight_pairs, dict):
        token_weight_pairs = token_weight_pairs[embedding_key]
    batch = []
    for seq in token_weight_pairs:
        batch.append([int(t[0]) if isinstance(t, (tuple, list)) else int(t) for t in seq])
    return torch.tensor(batch, dtype=torch.long)


class NunchakuKleinTEShim(nn.Module):
    """Duck-typed comfy ``cond_stage_model`` around a nunchaku Qwen3 text-encoder runtime."""

    def __init__(self, inner, layers=DEFAULT_LAYERS, pad_token: int = PAD_TOKEN,
                 embedding_key: str = "qwen3_8b"):
        super().__init__()
        self.inner = inner
        self.layers = tuple(layers)
        self.pad_token = pad_token
        self.embedding_key = embedding_key

    # comfy CLIP options plumbing (layer overrides don't apply here) -------------------------
    def set_clip_options(self, options):
        return None

    def reset_clip_options(self):
        return None

    # main entry used by CLIPTextEncode via comfy.sd.CLIP ------------------------------------
    def encode_token_weights(self, token_weight_pairs):
        device = next(self.inner.parameters()).device
        ids = tokens_to_tensor(token_weight_pairs, self.embedding_key).to(device)
        attention_mask = (ids != self.pad_token).long()

        with torch.no_grad():
            out = self.inner(
                input_ids=ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
        cond = assemble_klein_conditioning(out.hidden_states, self.layers).float().cpu()
        extra = {"attention_mask": attention_mask.cpu()}
        return cond, None, extra


class NunchakuQwen3TELoader:
    """Load the int4 (AWQ W4A16) Qwen3 text encoder as a FLUX.2 [klein] CLIP."""

    @classmethod
    def INPUT_TYPES(cls):
        try:
            import folder_paths
            files = folder_paths.get_filename_list("text_encoders")
        except Exception:
            files = []
        return {
            "required": {
                "te_name": (files, {"tooltip": "svdq-int4-Qwen3-text-Nunchaku.safetensors"}),
            }
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load"
    CATEGORY = "flux2/nunchaku"
    TITLE = "Nunchaku Qwen3 Text Encoder Loader (FLUX.2 klein)"

    def load(self, te_name: str):
        import comfy.model_management
        import comfy.sd
        import folder_paths
        from comfy.text_encoders.flux import KleinTokenizer8B

        try:
            from nunchaku.models.text_encoders.qwen_encoder import NunchakuQwenEncoderModel
        except ImportError as exc:
            raise ImportError(
                "This nunchaku build has no Qwen text-encoder runtime "
                "(nunchaku.models.text_encoders.qwen_encoder). Update the vitoom-nunchaku wheel."
            ) from exc

        path = folder_paths.get_full_path_or_raise("text_encoders", te_name)
        device = comfy.model_management.text_encoder_device()
        inner = NunchakuQwenEncoderModel.from_pretrained(path)
        inner = inner.to(device) if hasattr(inner, "to") else inner
        if hasattr(inner, "eval"):
            inner.eval()

        shim_holder = {}

        class _Target:
            class clip:  # noqa: N801 — comfy instantiates target.clip(device=..., dtype=..., model_options=...)
                def __new__(cls, device="cpu", dtype=None, model_options={}):
                    shim = NunchakuKleinTEShim(inner)
                    shim_holder["shim"] = shim
                    return shim

            tokenizer = KleinTokenizer8B

        params = sum(p.numel() for p in inner.parameters()) if hasattr(inner, "parameters") else 0
        clip = comfy.sd.CLIP(
            _Target,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            parameters=params,
        )
        logger.info("Nunchaku Qwen3 TE loaded from %s (%.1fM params visible)", te_name, params / 1e6)
        return (clip,)

from __future__ import annotations

import torch

from ..runtime.infer.seam_latent_anchor import apply_seam_anchor_correction, prepare_seam_anchor_state


class SeamLatentAnchorNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "anchor_latent": ("LATENT",),
                "mask": ("MASK",),
                "strength": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.5, "step": 0.05}),
                "anchor_width_px": ("INT", {"default": 3, "min": 1, "max": 256, "step": 1}),
                "anchor_falloff_px": ("INT", {"default": 16, "min": 1, "max": 512, "step": 1}),
                "process_left": ("BOOLEAN", {"default": True}),
                "process_right": ("BOOLEAN", {"default": True}),
                "process_top": ("BOOLEAN", {"default": True}),
                "process_bottom": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "ramp_curve": ("FLOAT", {"default": 1.5, "min": 0.5, "max": 8.0, "step": 0.1}),
                "profile_reduce": (["mean", "median"], {"default": "mean"}),
                "debug": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "seam"

    def apply(
        self,
        model,
        anchor_latent,
        mask,
        strength,
        anchor_width_px,
        anchor_falloff_px,
        process_left,
        process_right,
        process_top,
        process_bottom,
        ramp_curve=1.5,
        profile_reduce="mean",
        debug=False,
    ):
        if strength <= 0.0:
            return (model,)
        samples = anchor_latent["samples"].float()
        anchor_state = prepare_seam_anchor_state(
            samples,
            mask,
            anchor_width_px=int(anchor_width_px),
            anchor_falloff_px=int(anchor_falloff_px),
            process_left=bool(process_left),
            process_right=bool(process_right),
            process_top=bool(process_top),
            process_bottom=bool(process_bottom),
            reduce=str(profile_reduce),
        )
        if not anchor_state["sides"]:
            if debug:
                print("[SeamLatentAnchor] No valid seam context found; node inactive.")
            return (model,)

        _anchor_state = anchor_state
        _strength = float(strength)
        _curve = max(float(ramp_curve), 1e-3)
        _debug = bool(debug)
        _state = {
            "sigma_max": None,
            "last_sigma_logged": None,
            "step": 0,
        }

        def _seam_anchor_fn(args: dict) -> torch.Tensor:
            denoised = args["denoised"]
            sigma = args["sigma"]
            try:
                s = sigma.max().item()
            except Exception:
                s = float(sigma)
            if _state["sigma_max"] is None or s > _state["sigma_max"]:
                _state["sigma_max"] = s
                _state["step"] = 0
            sigma_max = _state["sigma_max"]
            sigma_progress = max(
                0.0,
                min(1.0, (sigma_max - s) / sigma_max if sigma_max > 1e-6 else 0.0),
            )
            _state["step"] += 1
            step_progress = 1.0 - 0.5 ** _state["step"]
            progress = 0.65 * step_progress + 0.35 * sigma_progress
            curved = 1.0 - (1.0 - progress) ** _curve
            effective = _strength * curved
            if effective < 1e-5:
                return denoised
            corrected = apply_seam_anchor_correction(
                denoised,
                _anchor_state,
                effective,
            )
            if _debug and s != _state["last_sigma_logged"]:
                _state["last_sigma_logged"] = s
                applied = (corrected - denoised).abs().mean().item()
                print(
                    f"[SeamLatentAnchor] step={_state['step']} sigma={s:.4f} "
                    f"progress={progress:.3f} effective={effective:.3f} "
                    f"weight_max={max(float(w.max().item()) for w in _anchor_state['weights'].values()):.3f} "
                    f"sides={','.join(_anchor_state['sides']) or 'none'} "
                    f"applied={applied:.5f}"
                )
            return corrected

        m = model.clone()
        m.model_options.setdefault("sampler_post_cfg_function", []).append(_seam_anchor_fn)
        return (m,)

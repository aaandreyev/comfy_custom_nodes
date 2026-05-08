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
                "topology_mask": ("MASK",),
                "ramp_curve": ("FLOAT", {"default": 1.5, "min": 0.5, "max": 8.0, "step": 0.1}),
                "profile_reduce": (["mean", "median"], {"default": "mean"}),
                "debug": ("BOOLEAN", {"default": False}),
                "low_freq_anchor_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "low_freq_anchor_decay_px": ("INT", {"default": 64, "min": 1, "max": 1024, "step": 1}),
                "safety_ring_px": ("INT", {"default": 0, "min": 0, "max": 512, "step": 1}),
            },
        }

    RETURN_TYPES = ("MODEL", "MASK")
    RETURN_NAMES = ("model", "generation_weight")
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
        topology_mask=None,
        ramp_curve=1.5,
        profile_reduce="mean",
        debug=False,
        low_freq_anchor_strength=0.0,
        low_freq_anchor_decay_px=64,
        safety_ring_px=0,
    ):
        samples = anchor_latent["samples"]
        if strength <= 0.0 and float(low_freq_anchor_strength) <= 0.0:
            zero_mask = torch.zeros(1, samples.shape[-2], samples.shape[-1])
            return (model, zero_mask)
        samples = samples.float()
        anchor_state = prepare_seam_anchor_state(
            samples,
            mask,
            topology_mask=topology_mask,
            anchor_width_px=int(anchor_width_px),
            anchor_falloff_px=int(anchor_falloff_px),
            process_left=bool(process_left),
            process_right=bool(process_right),
            process_top=bool(process_top),
            process_bottom=bool(process_bottom),
            reduce=str(profile_reduce),
            low_freq_anchor_decay_px=int(low_freq_anchor_decay_px),
            safety_ring_px=int(safety_ring_px),
        )
        if (
            not anchor_state["sides"]
            and not anchor_state.get("has_extra_contributions")
            and anchor_state.get("low_freq_target") is None
        ):
            if debug:
                print("[SeamLatentAnchor] No valid seam context found; node inactive.")
            gen_weight = anchor_state.get("generation_weight")
            if gen_weight is not None:
                gen_weight_out = gen_weight[:, 0, :, :]
            else:
                gen_weight_out = torch.zeros(1, samples.shape[-2], samples.shape[-1])
            return (model, gen_weight_out)

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
                low_freq_strength=float(low_freq_anchor_strength) * curved,
            )
            if _debug and s != _state["last_sigma_logged"]:
                _state["last_sigma_logged"] = s
                applied = (corrected - denoised).abs().mean().item()
                base_weight_max = max(
                    [float(w.max().item()) for w in _anchor_state["weights"].values()] or [0.0]
                )
                print(
                    f"[SeamLatentAnchor] step={_state['step']} sigma={s:.4f} "
                    f"progress={progress:.3f} effective={effective:.3f} "
                    f"weight_max={base_weight_max:.3f} "
                    f"sides={','.join(_anchor_state['sides']) or 'none'} "
                    f"present={','.join(_anchor_state.get('present_positions', ())) or 'auto'} "
                    f"applied={applied:.5f}"
                )
            return corrected

        m = model.clone()
        m.model_options.setdefault("sampler_post_cfg_function", []).append(_seam_anchor_fn)
        gen_weight = anchor_state.get("generation_weight")
        if gen_weight is not None:
            gen_weight_out = gen_weight[:, 0, :, :]  # [B, H, W]
        else:
            gen_weight_out = torch.zeros(1, samples.shape[-2], samples.shape[-1])
        return (m, gen_weight_out)

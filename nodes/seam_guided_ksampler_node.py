from __future__ import annotations

import math

import torch
from tqdm.auto import trange

import comfy.model_management
import comfy.samplers
import comfy.utils
import latent_preview

from ..runtime.infer.seam_latent_anchor import apply_seam_latent_guidance, prepare_seam_anchor_state


def _time_shift(mu, sigma, t):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def _get_lin_function(x1=256, y1=0.5, x2=4096, y2=1.15):
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def _get_schedule(num_steps, image_seq_len, base_shift=0.5, max_shift=1.15):
    timesteps = torch.linspace(1, 0, num_steps + 1)
    mu = _get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
    for i, t in enumerate(timesteps):
        tv = t.item()
        if 0 < tv < 1:
            timesteps[i] = _time_shift(mu, 1.0, tv)
    return timesteps.tolist()


def _seam_temporal_decay(step_index: int, total_steps: int, ramp_curve: float) -> float:
    if total_steps <= 1:
        return 1.0
    progress = step_index / max(total_steps - 1, 1)
    return max(0.0, 1.0 - progress) ** (1.0 / max(ramp_curve, 1e-3))


def _seam_temporal_rise(step_index: int, total_steps: int, ramp_curve: float) -> float:
    if total_steps <= 1:
        return 1.0
    progress = step_index / max(total_steps - 1, 1)
    return max(0.0, progress) ** (1.0 / max(ramp_curve, 1e-3))


def _windowed_progress(step_index: int, total_steps: int, start: float, end: float) -> float:
    if total_steps <= 1:
        return 1.0
    progress = step_index / max(total_steps - 1, 1)
    if progress <= start:
        return 0.0
    if progress >= end:
        return 1.0
    return (progress - start) / max(end - start, 1e-6)


def _clone_conditioning(conditioning, *, guidance_embed: float | None):
    if conditioning is None:
        return None
    cloned = []
    for item in conditioning:
        if item is None:
            cloned.append(None)
            continue
        tensor = item[0]
        meta = item[1].copy() if len(item) > 1 and isinstance(item[1], dict) else {}
        if guidance_embed is not None and "guidance" not in meta:
            meta["guidance"] = float(guidance_embed)
        if "ref_latents" in meta and "reference_latents" not in meta:
            meta["reference_latents"] = meta["ref_latents"]
        cloned.append([tensor, meta])
    return cloned


def _can_use_comfy_conditioning_pipeline(model) -> bool:
    return hasattr(model, "model_sampling")


class SeamGuidedKSamplerNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "anchor_latent": ("LATENT",),
                "mask": ("MASK",),
                "steps": ("INT", {"default": 4, "min": 1, "max": 200}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 30.0, "step": 0.1}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "base_shift": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.01}),
                "max_shift": ("FLOAT", {"default": 1.15, "min": 0.0, "max": 10.0, "step": 0.01}),
                "seam_noise_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.05}),
                "seam_noise_ramp_curve": ("FLOAT", {"default": 1.5, "min": 0.25, "max": 8.0, "step": 0.1}),
                "seam_start_step": ("INT", {"default": 1, "min": 1, "max": 200, "step": 1}),
                "seam_end_step": ("INT", {"default": 0, "min": 0, "max": 200, "step": 1}),
                "boundary_only_guidance": ("BOOLEAN", {"default": False}),
                "anchor_width_px": ("INT", {"default": 3, "min": 1, "max": 256, "step": 1}),
                "anchor_falloff_px": ("INT", {"default": 16, "min": 1, "max": 512, "step": 1}),
                "process_left": ("BOOLEAN", {"default": True}),
                "process_right": ("BOOLEAN", {"default": True}),
                "process_top": ("BOOLEAN", {"default": True}),
                "process_bottom": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "topology_mask": ("MASK",),
                "guidance_embed": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "profile_reduce": (["mean", "median"], {"default": "mean"}),
                "noise_mode": (["mean_shift", "matched_noise"], {"default": "mean_shift"}),
                "algorithm_mode": (["legacy", "phased_experimental"], {"default": "legacy"}),
                "debug": ("BOOLEAN", {"default": False}),
                "preserve_outside_latent": ("BOOLEAN", {"default": True}),
                "safety_ring_px": ("INT", {"default": 16, "min": 0, "max": 512, "step": 1}),
                "low_freq_anchor_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "low_freq_anchor_decay_px": ("INT", {"default": 64, "min": 1, "max": 1024, "step": 1}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "seam"

    def sample(
        self,
        model,
        positive,
        negative,
        latent_image,
        anchor_latent,
        mask,
        steps,
        seed,
        cfg,
        denoise,
        base_shift,
        max_shift,
        seam_noise_strength,
        seam_noise_ramp_curve,
        seam_start_step,
        seam_end_step,
        boundary_only_guidance,
        anchor_width_px,
        anchor_falloff_px,
        process_left,
        process_right,
        process_top,
        process_bottom,
        topology_mask=None,
        guidance_embed=1.0,
        profile_reduce="mean",
        noise_mode="mean_shift",
        algorithm_mode="legacy",
        debug=False,
        preserve_outside_latent=True,
        safety_ring_px=16,
        low_freq_anchor_strength=0.0,
        low_freq_anchor_decay_px=64,
    ):
        device = comfy.model_management.get_torch_device()
        latent = latent_image["samples"]
        reference_latent = latent.float()
        anchor_samples = anchor_latent["samples"].float()
        B, C, H, W = latent.shape

        anchor_state = prepare_seam_anchor_state(
            anchor_samples,
            mask,
            topology_mask=topology_mask,
            anchor_width_px=int(anchor_width_px),
            anchor_falloff_px=int(anchor_falloff_px),
            process_left=bool(process_left),
            process_right=bool(process_right),
            process_top=bool(process_top),
            process_bottom=bool(process_bottom),
            reduce=str(profile_reduce),
            safety_ring_px=int(safety_ring_px),
            low_freq_anchor_decay_px=int(low_freq_anchor_decay_px),
        )

        comfy.model_management.load_models_gpu([model])
        diffusion_model = model.model.diffusion_model
        model_dtype = diffusion_model.dtype

        patch_size = diffusion_model.patch_size
        h_tokens = (H + patch_size // 2) // patch_size
        w_tokens = (W + patch_size // 2) // patch_size
        image_seq_len = h_tokens * w_tokens

        schedule = _get_schedule(steps, image_seq_len, base_shift=base_shift, max_shift=max_shift)
        if denoise < 1.0:
            start_idx = 0
            for i, t in enumerate(schedule):
                if t <= denoise:
                    start_idx = max(0, i)
                    break
            schedule = schedule[start_idx:]

        generator = torch.Generator(device="cpu").manual_seed(seed)
        noise = torch.randn(latent.shape, generator=generator, dtype=torch.float32, device="cpu")
        if denoise < 1.0 and schedule:
            t_start = schedule[0]
            x = ((1.0 - t_start) * latent.float() + t_start * noise)
        else:
            x = noise
        x = x.to(device=device, dtype=model_dtype)
        latent_device = latent.to(device=device, dtype=model_dtype)
        ref_latent_device = reference_latent.to(device=device, dtype=model_dtype)
        noise_device = noise.to(device=device, dtype=model_dtype)
        generation_weight = None
        if bool(preserve_outside_latent):
            generation_weight = anchor_state.get("generation_weight")
            if generation_weight is not None:
                generation_weight = generation_weight.to(device=device, dtype=model_dtype)
                if generation_weight.shape[-2:] != x.shape[-2:]:
                    generation_weight = torch.nn.functional.interpolate(
                        generation_weight,
                        size=x.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    ).clamp(0.0, 1.0)

        use_cfg = cfg > 1.0 and negative is not None

        has_guidance_embed = getattr(diffusion_model.params, "guidance_embed", False)
        use_comfy_cond_pipeline = _can_use_comfy_conditioning_pipeline(model)
        positive_conds = None
        negative_conds = None
        model_options = model.model_options
        guidance_vec = None
        if has_guidance_embed:
            guidance_vec = torch.full((B,), guidance_embed, device=device, dtype=model_dtype)

        cond = positive[0][0].to(device=device, dtype=model_dtype)
        if cond.shape[0] != B:
            cond = cond[:1].expand(B, -1, -1)
        neg_cond = None
        ref_latents = None
        if use_comfy_cond_pipeline:
            processed = comfy.samplers.process_conds(
                model,
                x,
                {
                    "positive": _clone_conditioning(
                        positive,
                        guidance_embed=float(guidance_embed) if has_guidance_embed else None,
                    ),
                    **(
                        {
                            "negative": _clone_conditioning(
                                negative,
                                guidance_embed=float(guidance_embed) if has_guidance_embed else None,
                            )
                        }
                        if negative is not None else {}
                    ),
                },
                device,
                latent_image=latent_device,
                seed=seed,
            )
            positive_conds = processed["positive"]
            negative_conds = processed.get("negative")
        else:
            if use_cfg:
                neg_cond = negative[0][0].to(device=device, dtype=model_dtype)
                if neg_cond.shape[0] != B:
                    neg_cond = neg_cond[:1].expand(B, -1, -1)
            cond_meta = positive[0][1] if len(positive[0]) > 1 else {}
            for key in ("ref_latents", "reference_latents", "concat_latent_image"):
                if key in cond_meta:
                    ref_val = cond_meta[key]
                    if isinstance(ref_val, torch.Tensor):
                        ref_latents = [ref_val.to(device=device, dtype=model_dtype)]
                    elif isinstance(ref_val, list):
                        ref_latents = [r.to(device=device, dtype=model_dtype) for r in ref_val]
                    break
        total_steps = max(len(schedule) - 1, 0)
        pbar = comfy.utils.ProgressBar(total_steps)
        preview_callback = latent_preview.prepare_callback(model, total_steps)

        if debug:
            print(
                f"[SeamGuidedKSampler] steps={total_steps} denoise={denoise:.3f} cfg={cfg:.3f} "
                f"mode={noise_mode} seam_strength={seam_noise_strength:.3f} "
                f"sides={','.join(anchor_state['sides']) or 'none'} "
                f"present={','.join(anchor_state.get('present_positions', ())) or 'auto'} "
                f"boundary_only={bool(boundary_only_guidance)}"
            )

        active_start = max(1, int(seam_start_step))
        active_end = total_steps if int(seam_end_step) <= 0 else min(int(seam_end_step), total_steps)

        with torch.no_grad():
            for i in trange(total_steps, desc="Seam Guided KSampler"):
                comfy.model_management.throw_exception_if_processing_interrupted()

                t_curr = schedule[i]
                t_prev = schedule[i + 1]
                t_vec = torch.full((B,), t_curr, device=device, dtype=model_dtype)
                if use_comfy_cond_pipeline:
                    if use_cfg:
                        pred, pred_uncond = comfy.samplers.calc_cond_batch(
                            model,
                            [positive_conds, negative_conds],
                            x,
                            t_vec,
                            model_options,
                        )
                        pred = pred_uncond + cfg * (pred - pred_uncond)
                    else:
                        (pred,) = comfy.samplers.calc_cond_batch(
                            model,
                            [positive_conds],
                            x,
                            t_vec,
                            model_options,
                        )
                else:
                    transformer_options = model.model_options.get("transformer_options", {}).copy()
                    transformer_options["sigmas"] = t_vec
                    pred = diffusion_model.forward(
                        x,
                        t_vec,
                        cond,
                        y=None,
                        guidance=guidance_vec,
                        ref_latents=ref_latents,
                        control=None,
                        transformer_options=transformer_options,
                    )
                    if use_cfg:
                        pred_uncond = diffusion_model.forward(
                            x,
                            t_vec,
                            neg_cond,
                            y=None,
                            guidance=guidance_vec,
                            ref_latents=ref_latents,
                            control=None,
                            transformer_options=transformer_options,
                        )
                        pred = pred_uncond + cfg * (pred - pred_uncond)

                if preview_callback is not None:
                    x0_est = (x - t_curr * pred) if t_curr > 1e-6 else x
                    preview_callback(i, x0_est.cpu().float(), x.cpu().float(), total_steps)

                x = x + (t_prev - t_curr) * pred

                step_num = i + 1
                if algorithm_mode == "phased_experimental":
                    if noise_mode == "matched_noise":
                        phase = 1.0 - _windowed_progress(i, total_steps, 0.0, 0.6)
                        seam_temporal = max(0.0, phase) ** (1.0 / max(float(seam_noise_ramp_curve), 1e-3))
                    else:
                        phase = _windowed_progress(i, total_steps, 0.25, 1.0)
                        seam_temporal = max(0.0, phase) ** (1.0 / max(float(seam_noise_ramp_curve), 1e-3))
                    low_freq_phase = _windowed_progress(i, total_steps, 0.5, 1.0)
                    low_freq_temporal = max(0.0, low_freq_phase) ** (1.0 / max(float(seam_noise_ramp_curve), 1e-3))
                    match_contribution_variance = noise_mode == "matched_noise"
                else:
                    seam_temporal = _seam_temporal_decay(i, total_steps, float(seam_noise_ramp_curve))
                    low_freq_temporal = _seam_temporal_decay(i, total_steps, float(seam_noise_ramp_curve))
                    match_contribution_variance = False
                if (
                    (seam_noise_strength > 0.0 or float(low_freq_anchor_strength) > 0.0)
                    and (
                        anchor_state["sides"]
                        or anchor_state.get("extra_contributions")
                        or anchor_state.get("low_freq_target") is not None
                    )
                    and max(seam_temporal, low_freq_temporal) > 1e-5
                    and active_start <= step_num <= active_end
                ):
                    x = apply_seam_latent_guidance(
                        x,
                        anchor_state,
                        seam_noise_strength * seam_temporal,
                        mode=noise_mode,
                        boundary_only=bool(boundary_only_guidance),
                        low_freq_strength=float(low_freq_anchor_strength) * low_freq_temporal,
                        match_contribution_variance=match_contribution_variance,
                    )
                    if debug:
                        print(
                            f"[SeamGuidedKSampler] step={i + 1} sigma={t_curr:.4f} "
                            f"seam_temporal={seam_temporal:.3f} "
                            f"low_freq_temporal={low_freq_temporal:.3f} "
                            f"effective={seam_noise_strength * seam_temporal:.3f} "
                            f"mode={noise_mode}"
                        )
                if generation_weight is not None:
                    ref_x = (1.0 - t_prev) * ref_latent_device + t_prev * noise_device
                    x = x * generation_weight + ref_x * (1.0 - generation_weight)
                pbar.update(1)

        return ({"samples": x.cpu().float()},)

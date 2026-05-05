from __future__ import annotations

import math

import torch
from tqdm.auto import trange

import comfy.model_management
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


def _seam_temporal_strength(step_index: int, total_steps: int, ramp_curve: float) -> float:
    if total_steps <= 1:
        return 1.0
    progress = step_index / max(total_steps - 1, 1)
    return max(0.0, 1.0 - progress) ** (1.0 / max(ramp_curve, 1e-3))


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
                "guidance_embed": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "profile_reduce": (["mean", "median"], {"default": "mean"}),
                "noise_mode": (["mean_shift", "matched_noise"], {"default": "mean_shift"}),
                "debug": ("BOOLEAN", {"default": False}),
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
        guidance_embed=1.0,
        profile_reduce="mean",
        noise_mode="mean_shift",
        debug=False,
    ):
        device = comfy.model_management.get_torch_device()
        latent = latent_image["samples"]
        anchor_samples = anchor_latent["samples"].float()
        B, C, H, W = latent.shape

        anchor_state = prepare_seam_anchor_state(
            anchor_samples,
            mask,
            anchor_width_px=int(anchor_width_px),
            anchor_falloff_px=int(anchor_falloff_px),
            process_left=bool(process_left),
            process_right=bool(process_right),
            process_top=bool(process_top),
            process_bottom=bool(process_bottom),
            reduce=str(profile_reduce),
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

        cond = positive[0][0].to(device=device, dtype=model_dtype)
        if cond.shape[0] != B:
            cond = cond[:1].expand(B, -1, -1)

        neg_cond = None
        use_cfg = cfg > 1.0 and negative is not None
        if use_cfg:
            neg_cond = negative[0][0].to(device=device, dtype=model_dtype)
            if neg_cond.shape[0] != B:
                neg_cond = neg_cond[:1].expand(B, -1, -1)

        cond_meta = positive[0][1] if len(positive[0]) > 1 else {}
        ref_latents = None
        for key in ("ref_latents", "reference_latents", "concat_latent_image"):
            if key in cond_meta:
                ref_val = cond_meta[key]
                if isinstance(ref_val, torch.Tensor):
                    ref_latents = [ref_val.to(device=device, dtype=model_dtype)]
                elif isinstance(ref_val, list):
                    ref_latents = [r.to(device=device, dtype=model_dtype) for r in ref_val]
                break

        has_guidance_embed = getattr(diffusion_model.params, "guidance_embed", False)
        guidance_vec = None
        if has_guidance_embed:
            guidance_vec = torch.full((B,), guidance_embed, device=device, dtype=model_dtype)

        transformer_options = model.model_options.get("transformer_options", {}).copy()
        total_steps = max(len(schedule) - 1, 0)
        pbar = comfy.utils.ProgressBar(total_steps)
        preview_callback = latent_preview.prepare_callback(model, total_steps)

        if debug:
            print(
                f"[SeamGuidedKSampler] steps={total_steps} denoise={denoise:.3f} cfg={cfg:.3f} "
                f"mode={noise_mode} seam_strength={seam_noise_strength:.3f} "
                f"sides={','.join(anchor_state['sides']) or 'none'} "
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
                temporal = _seam_temporal_strength(i, total_steps, float(seam_noise_ramp_curve))
                if (
                    seam_noise_strength > 0.0
                    and anchor_state["sides"]
                    and temporal > 1e-5
                    and active_start <= step_num <= active_end
                ):
                    x = apply_seam_latent_guidance(
                        x,
                        anchor_state,
                        seam_noise_strength * temporal,
                        mode=noise_mode,
                        boundary_only=bool(boundary_only_guidance),
                    )
                    if debug:
                        print(
                            f"[SeamGuidedKSampler] step={i + 1} sigma={t_curr:.4f} "
                            f"temporal={temporal:.3f} effective={seam_noise_strength * temporal:.3f} "
                            f"mode={noise_mode}"
                        )
                pbar.update(1)

        return ({"samples": x.cpu().float()},)

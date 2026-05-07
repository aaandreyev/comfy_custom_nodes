from __future__ import annotations

import math

import torch
from tqdm.auto import trange

try:
    import comfy.model_management
    import comfy.samplers
    import comfy.utils
    import latent_preview
except ModuleNotFoundError:  # Optional in bare test environments without ComfyUI.
    comfy = None
    latent_preview = None

from ..runtime.infer.spatial_edit_denoise import (
    apply_spatial_denoise_preservation,
    build_continuous_schedule,
    build_local_denoise_state,
    normalize_mask,
)


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


class Flux2KleinSpatialDenoiseKSamplerNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "mask": ("MASK",),
                "steps": ("INT", {"default": 12, "min": 1, "max": 200}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 30.0, "step": 0.1}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "base_shift": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.01}),
                "max_shift": ("FLOAT", {"default": 1.15, "min": 0.0, "max": 10.0, "step": 0.01}),
                "local_denoise_enabled": ("BOOLEAN", {"default": True}),
                "band_width": ("INT", {"default": 16, "min": 1, "max": 1024, "step": 1}),
                "band_gradient_px": ("INT", {"default": 8, "min": 0, "max": 1024, "step": 1}),
                "band_denoise_min": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "band_denoise_max": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "merge_mode": (["max", "normalized_sum"], {"default": "max"}),
            },
            "optional": {
                "topology_mask": ("MASK",),
                "guidance_embed": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "preserve_outside_latent": ("BOOLEAN", {"default": True}),
                "topology_threshold": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.005}),
                "debug": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "sample"
    CATEGORY = "flux2_klein"

    def sample(
        self,
        model,
        positive,
        negative,
        latent_image,
        mask,
        steps,
        seed,
        cfg,
        denoise,
        base_shift,
        max_shift,
        local_denoise_enabled,
        band_width,
        band_gradient_px,
        band_denoise_min,
        band_denoise_max,
        merge_mode,
        topology_mask=None,
        guidance_embed=1.0,
        preserve_outside_latent=True,
        topology_threshold=0.01,
        debug=False,
    ):
        if comfy is None or latent_preview is None:
            raise ModuleNotFoundError("Flux2KleinSpatialDenoiseKSampler requires ComfyUI runtime modules")

        device = comfy.model_management.get_torch_device()
        latent = latent_image["samples"]
        B, _, H, W = latent.shape

        mask_t = normalize_mask(mask, (H, W))
        # GAP-2: mask batch must be 1 (broadcast) or exactly match latent batch
        if mask_t.shape[0] not in (1, B):
            raise ValueError(
                f"mask batch size {mask_t.shape[0]} is incompatible with latent batch size {B}; "
                "mask must have batch size 1 or exactly match the latent batch size"
            )

        denoise_state = build_local_denoise_state(
            mask_t,
            global_denoise=float(denoise),
            topology_mask=topology_mask,
            local_denoise_enabled=bool(local_denoise_enabled),
            band_width=int(band_width),
            band_gradient_px=int(band_gradient_px),
            band_denoise_min=float(band_denoise_min),
            band_denoise_max=float(band_denoise_max),
            merge_mode=str(merge_mode),
            topology_threshold=float(topology_threshold),
            preserve_outside_latent=bool(preserve_outside_latent),
        )

        comfy.model_management.load_models_gpu([model])
        diffusion_model = model.model.diffusion_model
        model_dtype = diffusion_model.dtype

        patch_size = diffusion_model.patch_size
        h_tokens = H // patch_size
        w_tokens = W // patch_size
        image_seq_len = h_tokens * w_tokens

        target_denoise_map = denoise_state["target_denoise_map"].to(dtype=torch.float32)
        effective_denoise = float(target_denoise_map.max().item()) if bool(local_denoise_enabled) else float(denoise)
        full_schedule = _get_schedule(steps, image_seq_len, base_shift=base_shift, max_shift=max_shift)
        schedule = build_continuous_schedule(
            full_schedule,
            effective_denoise,
            schedule_builder=lambda total_steps: _get_schedule(
                total_steps,
                image_seq_len,
                base_shift=base_shift,
                max_shift=max_shift,
            ),
        )
        start_sigma = float(schedule[0]) if schedule else 0.0

        # BUG-1: Scale per-pixel denoise fractions into schedule sigma space.
        # target_denoise_map ∈ [0, effective_denoise]; sigma_map ∈ [0, start_sigma].
        # Without this, local-denoise pixels are initialized at the wrong noise level
        # relative to the time-shifted schedule start (e.g. effective_denoise=0.5 maps
        # to start_sigma≈0.76 after time-shift, so raw target values under-noise).
        sigma_scale = start_sigma / effective_denoise if effective_denoise > 1e-6 else 1.0
        sigma_map = (target_denoise_map * sigma_scale).clamp(0.0, 1.0)

        generator = torch.Generator(device="cpu").manual_seed(seed)
        noise = torch.randn(latent.shape, generator=generator, dtype=torch.float32, device="cpu")
        if schedule and start_sigma > 1e-6:
            x = (1.0 - sigma_map) * latent.float() + sigma_map * noise
        else:
            x = latent.float().clone()
        x = x.to(device=device, dtype=model_dtype)
        latent_device = latent.to(device=device, dtype=model_dtype)
        noise_device = noise.to(device=device, dtype=model_dtype)
        sigma_map_device = sigma_map.to(device=device, dtype=model_dtype)

        use_cfg = cfg > 1.0 and negative is not None
        has_guidance_embed = getattr(diffusion_model.params, "guidance_embed", False)
        model_options = model.model_options

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
                    if negative is not None
                    else {}
                ),
            },
            device,
            latent_image=latent_device,
            seed=seed,
        )
        positive_conds = processed["positive"]
        negative_conds = processed.get("negative")

        total_steps = max(len(schedule) - 1, 0)
        # M-3: guard against 0-step degenerate case (denoise=0 or empty schedule)
        pbar = comfy.utils.ProgressBar(total_steps) if total_steps > 0 else None
        preview_callback = latent_preview.prepare_callback(model, total_steps) if total_steps > 0 else None

        if debug:
            fallback_count = sum(
                1 for s in denoise_state.get("per_sample", []) if s.get("uniform_fallback")
            )
            print(
                f"[Flux2KleinSpatialDenoiseKSampler] steps={total_steps} "
                f"denoise={denoise:.4f} effective_denoise={effective_denoise:.4f} "
                f"start_sigma={start_sigma:.4f} sigma_scale={sigma_scale:.4f} "
                f"local={bool(local_denoise_enabled)} "
                f"band_width={int(band_width)} gradient={int(band_gradient_px)} "
                f"band_min={float(band_denoise_min):.3f} band_max={float(band_denoise_max):.3f} "
                f"merge={merge_mode} present={','.join(denoise_state.get('present_positions', ())) or 'auto'}"
                + (f" uniform_fallback={fallback_count}/{B}" if fallback_count else "")
            )

        with torch.no_grad():
            for i in trange(total_steps, desc="Flux2KleinSpatialDenoiseKSampler"):
                comfy.model_management.throw_exception_if_processing_interrupted()

                t_curr = float(schedule[i])
                t_prev = float(schedule[i + 1])
                t_vec = torch.full((B,), t_curr, device=device, dtype=model_dtype)

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

                if preview_callback is not None:
                    x0_est = (x - t_curr * pred) if t_curr > 1e-6 else x
                    preview_callback(i, x0_est.cpu().float(), x.cpu().float(), total_steps)

                x = x + (t_prev - t_curr) * pred
                x = apply_spatial_denoise_preservation(
                    x,
                    latent_device,
                    noise_device,
                    sigma_map_device,
                    t_prev,
                )
                if pbar is not None:
                    pbar.update(1)

        return ({"samples": x.cpu().float()},)

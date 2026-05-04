from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from ..runtime.model_loader import available_checkpoints, load_model, pick_inference_device
from ..runtime.infer.correct_full_frame import apply_corrector_to_full_frame
from ..runtime.strip_ops import mask_bbox


class SeamHarmonizerV3Node:
    STREAM_KEYS = ("gain", "gamma", "bias", "mix", "detail", "gate", "attention")

    @classmethod
    def INPUT_TYPES(cls):
        model_choices = cls._model_choices()
        return {
            "required": {
                "IMAGE": ("IMAGE",),
                "MASK": ("MASK",),
                "model_name": (model_choices, {"default": cls._default_model_name(model_choices)}),
                "inner_width": ("INT", {"default": 128}),
                "inner_falloff_px": ("INT", {"default": 48, "min": 0, "max": 128, "step": 1}),
                "strength": ("FLOAT", {"default": 1.0}),
                "process_left": ("BOOLEAN", {"default": True}),
                "process_right": ("BOOLEAN", {"default": True}),
                "process_top": ("BOOLEAN", {"default": True}),
                "process_bottom": ("BOOLEAN", {"default": True}),
                "gain_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
                "gamma_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
                "bias_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
                "mix_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
                "detail_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
                "gate_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
                "attention_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
                "debug_previews": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "seam"

    @classmethod
    def _model_choices(cls) -> list[str]:
        choices = [
            candidate
            for candidate in available_checkpoints()
            if candidate.endswith((".safetensors", ".ckpt", ".pt", ".pth"))
        ]
        return choices or ["seam_harmonizer_v3.safetensors"]

    @classmethod
    def _default_model_name(cls, choices: list[str] | None = None) -> str:
        model_choices = choices or cls._model_choices()
        for candidate in model_choices:
            if candidate.endswith(".safetensors"):
                return candidate
        return model_choices[0]

    @staticmethod
    def _debug_root() -> Path:
        return Path.cwd() / "outputs" / "debug_previews"

    def run(
        self,
        IMAGE,
        MASK,
        model_name,
        inner_width,
        inner_falloff_px,
        strength,
        process_left,
        process_right,
        process_top,
        process_bottom,
        gain_strength,
        gamma_strength,
        bias_strength,
        mix_strength,
        detail_strength,
        gate_strength,
        attention_strength,
        debug_previews,
    ):
        device = pick_inference_device()
        model, _sidecar = load_model(model_name, device=device)
        image_bchw = IMAGE.permute(0, 3, 1, 2).contiguous()
        rgb = image_bchw[:, :3]
        alpha = image_bchw[:, 3:] if image_bchw.shape[1] > 3 else None
        mask = MASK.unsqueeze(1).float()
        original_mask_shape = [int(mask.shape[-2]), int(mask.shape[-1])]
        image_hw = [int(rgb.shape[-2]), int(rgb.shape[-1])]
        mask_resized = False
        if mask.shape[-2:] != rgb.shape[-2:]:
            mask = F.interpolate(mask, size=rgb.shape[-2:], mode="nearest")
            mask_resized = True
        mask = (mask > 0.5).float()
        bbox = mask_bbox(mask)
        x0, y0, x1, y1 = bbox
        base_meta = {
            "bbox": [x0, y0, x1, y1],
            "mask_mean": float(mask.mean().item()),
            "inner_width": int(inner_width),
            "inner_falloff_px": int(inner_falloff_px),
            "strength": float(strength),
            "model_name": str(model_name),
            "original_mask_shape": original_mask_shape,
            "image_shape_hw": image_hw,
            "mask_resized_to_image": mask_resized,
        }
        stream_strengths = {
            "gain": float(gain_strength),
            "gamma": float(gamma_strength),
            "bias": float(bias_strength),
            "mix": float(mix_strength),
            "detail": float(detail_strength),
            "gate": float(gate_strength),
            "attention": float(attention_strength),
        }
        base_meta["stream_strengths"] = stream_strengths
        sides = []
        if process_left and x0 > 0:
            sides.append("left")
        if process_right and rgb.shape[-1] - x1 > 0:
            sides.append("right")
        if process_top and y0 > 0:
            sides.append("top")
        if process_bottom and rgb.shape[-2] - y1 > 0:
            sides.append("bottom")
        if not sides:
            if debug_previews:
                self._write_debug({"per_side": {}, "reason": "no_processable_sides"}, rgb, rgb, extra={**base_meta, "sides": sides})
            return (IMAGE,)
        corrected_rgb, debug = apply_corrector_to_full_frame(
            model,
            rgb,
            mask,
            bbox,
            sides,
            inner_width,
            strength,
            int(inner_falloff_px),
            stream_strengths,
        )
        corrected = torch.cat((corrected_rgb, alpha), dim=1) if alpha is not None else corrected_rgb

        if debug_previews:
            self._write_debug(debug, rgb, corrected_rgb, extra={**base_meta, "sides": sides})
        return (corrected.permute(0, 2, 3, 1).contiguous(),)

    def _write_debug(self, debug: dict, image: torch.Tensor, corrected: torch.Tensor, extra: dict | None = None) -> None:
        root = self._debug_root() / datetime.now().strftime("%Y%m%d_%H%M%S")
        root.mkdir(parents=True, exist_ok=True)
        self._save_tensor(image[0], root / "input.png")
        self._save_tensor(corrected[0], root / "corrected.png")
        diff = (corrected - image).abs()
        self._save_tensor((diff[0] / diff[0].amax().clamp_min(1e-6)), root / "diff.png")
        merged = debug.get("merged_delta")
        if merged is not None:
            self._save_tensor((merged[0] + 0.5).clamp(0.0, 1.0), root / "merged_delta.png")
        for side, delta in debug.get("side_deltas", {}).items():
            self._save_tensor((delta[0] + 0.5).clamp(0.0, 1.0), root / f"side_{side}_delta.png")
        for side, confidence in debug.get("side_confidences", {}).items():
            self._save_tensor(confidence[0].repeat(3, 1, 1), root / f"confidence_{side}.png")
        for side, weight in debug.get("weights", {}).items():
            self._save_tensor(weight[0].repeat(3, 1, 1), root / f"weight_map_{side}.png")
        for key in ("gain_lowres", "gamma_lowres", "gate_lowres"):
            value = debug.get(key)
            if isinstance(value, torch.Tensor):
                for i in range(min(value.shape[0], 4)):
                    preview = value[i].repeat(3, 1, 1)
                    preview = (preview - preview.min()) / (preview.max() - preview.min()).clamp_min(1e-6)
                    self._save_tensor(preview, root / f"{key}_{i}.png")
        summary = {
            "per_side": debug.get("per_side", {}),
            "reason": debug.get("reason", "applied"),
            "debug_root": str(root),
            "max_abs_change": float(diff.max().item()),
            "mean_abs_change": float(diff.mean().item()),
        }
        if extra:
            summary.update(extra)
            bbox = extra.get("bbox")
            if isinstance(bbox, list) and len(bbox) == 4:
                mask_preview = torch.zeros(3, image.shape[-2], image.shape[-1], device=image.device, dtype=image.dtype)
                x0, y0, x1, y1 = [int(v) for v in bbox]
                mask_preview[:, y0:y1, x0:x1] = 1.0
                self._save_tensor(mask_preview, root / "mask.png")
        (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    @staticmethod
    def _save_tensor(x: torch.Tensor, path: Path) -> None:
        arr = (x.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).astype("uint8")
        Image.fromarray(arr).save(path)

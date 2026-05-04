from __future__ import annotations

import inspect

import torch

from ..data.harmonizer_input import build_harmonizer_input
from .extract_strips import extract_active_strips
from .merge_bands import merge_side_deltas


def _model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def _canonical_model_input(strip_batch: torch.Tensor, outer_width: int, boundary_band_px: int = 24) -> torch.Tensor:
    return build_harmonizer_input(strip_batch, outer_width=outer_width, boundary_band_px=boundary_band_px, seam_x=outer_width)["input"]


def _place_inner_map(
    canonical_inner: torch.Tensor,
    image: torch.Tensor,
    bbox: tuple[int, int, int, int],
    side: str,
    inner_width: int,
    meta: dict,
) -> torch.Tensor:
    placed = torch.zeros(image.shape[0], canonical_inner.shape[1], image.shape[-2], image.shape[-1], device=image.device, dtype=canonical_inner.dtype)
    x0, y0, x1, y1 = bbox
    if side == "left":
        width = min(inner_width, x1 - x0)
        y_start = int(meta["y_start"])
        y_end = min(y_start + canonical_inner.shape[-2], image.shape[-2])
        placed[:, :, y_start:y_end, x0 : x0 + width] = canonical_inner[:, :, : y_end - y_start, :width]
    elif side == "right":
        width = min(inner_width, x1 - x0)
        y_start = int(meta["y_start"])
        y_end = min(y_start + canonical_inner.shape[-2], image.shape[-2])
        placed[:, :, y_start:y_end, x1 - width : x1] = torch.flip(canonical_inner[:, :, : y_end - y_start, :width], dims=(-1,))
    elif side == "top":
        height = min(inner_width, y1 - y0)
        top_map = torch.rot90(canonical_inner[:, :, :, :height], k=3, dims=(-2, -1))
        x_start = int(meta["x_start"])
        x_end = min(x_start + top_map.shape[-1], image.shape[-1])
        placed[:, :, y0 : y0 + height, x_start:x_end] = top_map[:, :, :height, : x_end - x_start]
    elif side == "bottom":
        height = min(inner_width, y1 - y0)
        bottom_map = torch.rot90(canonical_inner[:, :, :, :height], k=1, dims=(-2, -1))
        x_start = int(meta["x_start"])
        x_end = min(x_start + bottom_map.shape[-1], image.shape[-1])
        placed[:, :, y1 - height : y1, x_start:x_end] = bottom_map[:, :, :height, : x_end - x_start]
    else:
        raise ValueError(f"unsupported side: {side}")
    return placed


def apply_corrector_to_full_frame(
    model: torch.nn.Module,
    image: torch.Tensor,
    mask: torch.Tensor,
    bbox: tuple[int, int, int, int],
    sides: list[str],
    inner_width: int,
    strength: float = 1.0,
    blend_falloff_px: int | None = None,
    stream_strengths: dict[str, float] | None = None,
) -> tuple[torch.Tensor, dict]:
    outputs = extract_active_strips(image[0], bbox, sides, inner_width)
    side_order = list(outputs.keys())
    if not side_order:
        return image, {"per_side": {}, "weights": {}, "side_deltas": {}, "merged_delta": torch.zeros_like(image)}

    outer_width = int(getattr(model, "outer_width", 128))
    strip_batch = torch.stack([outputs[side]["strip"] for side in side_order], dim=0)
    boundary_band_px = int(getattr(model, "boundary_band_px", 24))
    model_in = _canonical_model_input(strip_batch, outer_width, boundary_band_px=boundary_band_px).to(_model_device(model))
    # Any explicit UI value means "use geometric blending semantics from the node".
    # That includes 0, which now means a flat 1.0 band without fade.
    disable_spatial = blend_falloff_px is not None

    with torch.inference_mode():
        forward_sig = inspect.signature(model.forward).parameters
        if "disable_spatial_gate" in forward_sig or "stream_strengths" in forward_sig:
            kwargs = {}
            if "disable_spatial_gate" in forward_sig:
                kwargs["disable_spatial_gate"] = disable_spatial
            if "stream_strengths" in forward_sig:
                kwargs["stream_strengths"] = stream_strengths
            model_out = model(model_in, **kwargs)
        else:
            model_out = model(model_in)
    if not isinstance(model_out, dict):
        raise RuntimeError("SeamHarmonizerV3 inference requires dict outputs with corrected_inner")

    inner_delta = (model_out["corrected_inner"] - model_in[:, :3, :, outer_width:]).cpu()
    inner_confidence = model_out["confidence"].detach().cpu()
    debug = {"per_side": {}, "architecture": "seam_harmonizer_v3"}
    for key in ("gain_lowres", "gamma_lowres", "bias_lowres", "detail_lowres", "gate_lowres"):
        debug[key] = model_out[key].detach().cpu()

    side_deltas: dict[str, torch.Tensor] = {}
    side_confidences: dict[str, torch.Tensor] = {}
    for index, side in enumerate(side_order):
        delta_inner = inner_delta[index : index + 1]
        confidence_inner = inner_confidence[index : index + 1]
        meta = outputs[side]["meta"]
        side_deltas[side] = _place_inner_map(delta_inner.to(image.device), image, bbox, side, inner_width, meta)
        side_confidences[side] = _place_inner_map(confidence_inner.to(image.device), image, bbox, side, inner_width, meta)
        debug["per_side"][side] = {
            "edge_padded_pixels": outputs[side]["meta"]["edge_padded_pixels"],
            "confidence_mean": float(confidence_inner.mean().item()),
            "delta_abs_mean": float(delta_inner.abs().mean().item()),
        }

    merged, weights = merge_side_deltas(
        side_deltas,
        mask,
        side_confidences=side_confidences,
        bbox=bbox,
        inner_width=inner_width,
        blend_falloff_px=blend_falloff_px,
    )
    corrected = (image + merged * strength).clamp(0.0, 1.0)
    corrected = corrected * mask + image * (1.0 - mask)
    debug["weights"] = weights
    debug["side_deltas"] = side_deltas
    debug["side_confidences"] = side_confidences
    debug["merged_delta"] = merged
    return corrected, debug

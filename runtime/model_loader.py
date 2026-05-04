from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from .models.factory import build_model_from_config


_MODEL_CACHE: dict[tuple[str, str, float], tuple[torch.nn.Module, dict]] = {}


def pick_inference_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def available_checkpoints() -> list[str]:
    names: list[str] = []
    try:
        import folder_paths  # type: ignore

        names = list(folder_paths.get_filename_list("checkpoints"))
    except Exception:
        root = checkpoints_dir()
        if root.exists():
            names = sorted(
                str(path.relative_to(root))
                for path in root.rglob("*")
                if path.is_file() and path.suffix in {".safetensors", ".ckpt", ".pt", ".pth"}
            )
    return names


def checkpoints_dir() -> Path:
    try:
        import folder_paths  # type: ignore

        paths = folder_paths.get_folder_paths("checkpoints")
        if paths:
            return Path(paths[0])
    except Exception:
        pass

    for parent in Path(__file__).resolve().parents:
        candidate = parent / "models" / "checkpoints"
        if candidate.exists():
            return candidate
        if parent.name == "custom_nodes":
            comfy_candidate = parent.parent / "models" / "checkpoints"
            if comfy_candidate.exists():
                return comfy_candidate
    return Path("models") / "checkpoints"


def _resolve_model_path(model_name: str) -> Path:
    model_path = Path(model_name).expanduser()
    if model_path.is_absolute():
        return model_path

    try:
        import folder_paths  # type: ignore

        resolved = folder_paths.get_full_path("checkpoints", model_name)
        if resolved:
            return Path(resolved)
    except Exception:
        pass

    candidate = checkpoints_dir() / model_path
    if candidate.exists():
        return candidate
    return candidate


def _filter_matching_state_dict(
    model: torch.nn.Module,
    state: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], list[str], list[str]]:
    current = model.state_dict()
    matched: dict[str, torch.Tensor] = {}
    dropped_unexpected: list[str] = []
    dropped_mismatch: list[str] = []
    for key, value in state.items():
        if key not in current:
            dropped_unexpected.append(key)
            continue
        if not isinstance(value, torch.Tensor) or current[key].shape != value.shape:
            dropped_mismatch.append(key)
            continue
        matched[key] = value
    return matched, dropped_unexpected, dropped_mismatch


def load_model(model_name: str, device: str = "cpu") -> tuple[torch.nn.Module, dict]:
    model_path = _resolve_model_path(model_name)
    mtime = model_path.stat().st_mtime if model_path.exists() else 0.0
    key = (str(model_path), device, mtime)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            f"Expected a checkpoint under {checkpoints_dir()}."
        )
    sidecar_path = model_path.with_suffix(".json")
    if not sidecar_path.exists():
        raise FileNotFoundError(
            f"Sidecar JSON not found: {sidecar_path}. "
            "The .safetensors export must be next to a same-name .json file."
        )
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    _validate_sidecar(sidecar)
    state = load_file(str(model_path), device=device)
    model = build_model_from_config(sidecar)
    compatible_state, dropped_unexpected, dropped_mismatch = _filter_matching_state_dict(model, state)
    result = model.load_state_dict(compatible_state, strict=False)
    if result.missing_keys or dropped_unexpected or dropped_mismatch:
        print(
            f"[seam_harmonizer] partial load: missing={list(result.missing_keys)[:6]} "
            f"dropped_unexpected={dropped_unexpected[:6]} "
            f"dropped_mismatch={dropped_mismatch[:6]}",
            flush=True,
        )
    model.eval().to(device)
    _MODEL_CACHE[key] = (model, sidecar)
    return model, sidecar


def _validate_sidecar(sidecar: dict) -> None:
    if sidecar["schema_version"] != 1:
        raise RuntimeError("Unsupported schema_version")
    if sidecar["architecture"]["in_channels"] != 9:
        raise RuntimeError("Model must have 9 input channels")
    if sidecar["architecture"]["name"] != "seam_harmonizer_v3":
        raise RuntimeError("Only seam_harmonizer_v3 exports are supported")

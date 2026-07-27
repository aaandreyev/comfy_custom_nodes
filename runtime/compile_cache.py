"""Mega-cache: hot save/load of torch.compile artifacts inside one process.

``torch.compiler.save_cache_artifacts`` serializes every compile artifact the
current process has produced (FX graphs, autotune results, AOT artifacts) into
one blob; ``load_cache_artifacts`` hot-loads such a blob into the in-memory
caches of a fresh process, skipping most of the per-shape dynamo/inductor work
that even a warm on-disk inductor cache still pays on first use.

Both calls only make sense inside the ComfyUI process, so the pack loads a
blob at import time when ``COMFY_MEGA_CACHE`` points to one, and exposes
``POST /prefill_harmonization/save_mega_cache`` to serialize after a warmup.
Blobs are tied to the torch version and GPU architecture; a mismatched blob is
rejected gracefully.
"""
from __future__ import annotations

import os
import tempfile


def _info_summary(info) -> dict:
    out = {}
    for name in (
        "inductor_artifacts",
        "autotune_artifacts",
        "aot_autograd_artifacts",
        "pgo_artifacts",
        "precompile_artifacts",
    ):
        value = getattr(info, name, None)
        if value is None:
            continue
        try:
            out[name] = len(value)
        except TypeError:
            out[name] = str(value)
    return out or {"repr": str(info)[:200]}


def save_mega_cache(path: str) -> dict:
    import torch

    result = torch.compiler.save_cache_artifacts()
    if not result:
        return {"saved": False, "reason": "no compile artifacts in this process yet"}
    blob, info = result
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, suffix=".megacache.tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(blob)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return {"saved": True, "path": path, "bytes": len(blob), "info": _info_summary(info)}


def load_mega_cache(path: str) -> dict:
    import torch

    if not path or not os.path.exists(path):
        return {"loaded": False, "reason": f"no blob at {path!r}"}
    with open(path, "rb") as f:
        blob = f.read()
    try:
        info = torch.compiler.load_cache_artifacts(blob)
    except Exception as exc:
        return {"loaded": False, "bytes": len(blob), "reason": f"{type(exc).__name__}: {exc}"}
    if info is None:
        return {"loaded": False, "bytes": len(blob),
                "reason": "torch rejected the blob (torch version / GPU arch mismatch?)"}
    return {"loaded": True, "bytes": len(blob), "info": _info_summary(info)}

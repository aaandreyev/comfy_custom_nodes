"""Tests for the mega-cache save/load helpers."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT.parent))

from comfy_custom_nodes_repo.runtime.compile_cache import load_mega_cache, save_mega_cache


def test_load_missing_blob_is_graceful(tmp_path: Path) -> None:
    result = load_mega_cache(str(tmp_path / "absent.bin"))
    assert result["loaded"] is False
    assert "no blob" in result["reason"]


def test_load_garbage_blob_is_graceful(tmp_path: Path) -> None:
    blob = tmp_path / "garbage.bin"
    blob.write_bytes(b"definitely not a torch cache artifact bundle")
    result = load_mega_cache(str(blob))
    assert result["loaded"] is False


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    def fn(x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.relu(x * 2.0 + 1.0)

    try:
        compiled = torch.compile(fn)
        compiled(torch.randn(8, 8))
    except Exception as exc:
        pytest.skip(f"torch.compile unavailable in this environment: {exc}")

    path = tmp_path / "mega.bin"
    result = save_mega_cache(str(path))
    if not result["saved"]:
        pytest.skip(f"no artifacts to save: {result}")
    assert path.exists() and path.stat().st_size == result["bytes"] > 0

    loaded = load_mega_cache(str(path))
    assert loaded["loaded"] is True
    assert loaded["bytes"] == result["bytes"]

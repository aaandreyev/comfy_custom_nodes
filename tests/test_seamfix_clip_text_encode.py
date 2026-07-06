from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT.parent))

from comfy_custom_nodes_repo.nodes.seamfix_clip_text_encode_node import apply_seamfix_prefix


def test_apply_seamfix_prefix_adds_banner() -> None:
    assert apply_seamfix_prefix("a cat", True) == "SEAMFIX\na cat"


def test_apply_seamfix_prefix_disabled() -> None:
    assert apply_seamfix_prefix("a cat", False) == "a cat"


def test_apply_seamfix_prefix_idempotent_when_first_line_seamfix() -> None:
    assert apply_seamfix_prefix("SEAMFIX\na cat", True) == "SEAMFIX\na cat"


def test_apply_seamfix_empty() -> None:
    assert apply_seamfix_prefix("", True) == ""

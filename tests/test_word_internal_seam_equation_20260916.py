from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "word_internal_seam_equation_20260916",
    ROOT / "experiments/word_internal_seam_equation_20260916.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_preflight_rejects_overlap_before_rendering():
    result = MODULE.novelty_preflight()
    assert result["status"] == "passed"
    assert not result["signature_overlaps"]
    assert "fixed-tape" in result["rejected_routes"]


def test_word_internal_seam_run_emits_intact_long_scene_and_independent_audit():
    result = MODULE.run()
    row = result["best"]
    assert result["candidate_count"] == 9
    assert row["audit"]["letters"] > 100
    assert row["audit"]["independent_two_pointer_exact"] == row["audit"]["exact"]
    assert row["audit"]["sha256_forward"] != row["audit"]["sha256_reverse"]
    assert row["seam_match"] is True
    assert row["anti_shortcut_flags"]["fixed_tape"] is False
    assert row["anti_shortcut_flags"]["word_order_mirror"] is False
    assert "agent" in row["semantic_roles"] and "setting" in row["semantic_roles"]

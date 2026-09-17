from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("luna_cfg_exact_intersection_20260917", ROOT / "experiments/luna_cfg_exact_intersection_20260917.py")
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_preflight_is_before_search_and_ignores_self_collision():
    result = MODULE.novelty_preflight()
    assert result["status"] == "passed"
    assert result["ignored_self_registry_collision"] is True
    assert "fixed tape" in result["rejected_routes"]


def test_exact_rendered_candidate_has_independent_audits_and_chart():
    result = MODULE.run()
    row = result["best"]
    assert result["candidate_count"] == 1
    assert row["audit"]["letters"] > 38
    assert row["audit"]["exact"] is True
    assert row["audit"]["independent_two_pointer_exact"] is True
    assert row["audit"]["sha256_forward"] == row["audit"]["sha256_reverse"]
    assert row["chart_states"]
    assert result["grammar"]["chart_key"] == ["grammar_state", "left_position", "right_position"]
    assert row["anti_shortcut_flags"]["fixed_tape"] is False
    assert row["anti_shortcut_flags"]["word_order_mirror"] is True
    assert row["anti_shortcut_flags"]["self_palindromic_span"] is True
    assert row["mechanically_admitted"] is False
    assert row["provenance"]["posthoc_resegmentation"] is False

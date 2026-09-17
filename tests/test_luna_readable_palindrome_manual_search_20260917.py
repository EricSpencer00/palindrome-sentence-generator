import importlib.util
from pathlib import Path

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("manual_lane", ROOT / "experiments" / "luna_readable_palindrome_manual_search_20260917.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_manual_lane_keeps_complete_intact_prose_and_independent_checks():
    result = MODULE.run()
    assert result["novelty_preflight"]["passed"]
    assert result["candidate_count"] > 0
    assert result["exact_count"] == 0
    row = result["best_intact_prose"]
    assert row["intact_prose"]
    assert row["letters"] > 38
    assert row["independent_exact_agreement"]
    assert row["next_repair"]
    assert not any(row["anti_shortcut_flags"].values())


def test_manual_inventory_is_fresh_and_role_bearing():
    assert len(MODULE.SUBJECTS) >= 6
    assert len(MODULE.VERBS) >= 6
    assert all(role for _, role in MODULE.SUBJECTS + MODULE.VERBS + MODULE.OBJECTS + MODULE.ADVERBS)
    assert MODULE.novelty_preflight()["collisions"] == []

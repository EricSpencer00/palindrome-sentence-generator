import importlib.util
from pathlib import Path

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("fresh_center", ROOT / "experiments" / "luna_fresh_center_composition_20260917.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_fresh_center_composition_has_complete_prose_and_live_residuals():
    result = MODULE.run()
    assert result["novelty_preflight"]["passed"]
    assert result["center_event"]["text"] == "the bell rings"
    assert result["candidate_count"] == 12
    assert result["exact_count"] == 0
    for row in result["rendered_candidates"]:
        assert row["complete_grammar"]
        assert row["independent_exact_agreement"]
        assert row["obligations"]["pairs_checked"] > 38
        assert row["residual_repair"] is not None
        assert not any(row["anti_shortcut_flags"].values())
        assert row["next_reader_facing_test"]


def test_center_is_not_self_palindromic_or_repeated_and_novel():
    assert MODULE.letters(MODULE.CENTER_EVENT["text"]) != MODULE.letters(MODULE.CENTER_EVENT["text"])[::-1]
    assert MODULE.novelty_preflight()["collisions"] == []

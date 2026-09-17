import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("luna_scene", ROOT / "experiments" / "luna_palindromic_scene_realizer_20260917.py")
LANE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LANE)


def test_scene_realizer_records_complete_typed_prose_and_independent_audits():
    result = LANE.run()
    assert result["stats"]["rendered"] == 3
    assert result["stats"]["exact_over_38"] == 0
    assert all(row["complete_svo_pp"] for row in result["candidates"])
    assert all(row["audit"]["letters"] > 38 for row in result["candidates"])
    assert all(row["audit"]["independent_pointer_exact"] is False for row in result["candidates"])
    assert all(row["audit"]["normalized_sha256"] != row["audit"]["reverse_sha256"] for row in result["candidates"])


def test_scene_realizer_has_no_fixed_tape_and_repairs_each_failure():
    result = LANE.run()
    assert result["novelty_preflight"]["fixed_tape_used"] is False
    assert result["novelty_preflight"]["self_collision_ignored"] is False or result["novelty_preflight"]["output_excluded"]
    for row in result["candidates"]:
        assert row["audit"]["anti_shortcut"]["mirrored_word_order"] is False
        assert row["audit"]["repair"]

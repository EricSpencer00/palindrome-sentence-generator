import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("semantic_relation_plan_solver_20260916", ROOT / "experiments/semantic_relation_plan_solver_20260916.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_novelty_preflight_is_new_and_relation_state_is_typed():
    preflight = MODULE.novelty_preflight()
    assert preflight["passed"]
    assert len(MODULE.RELATIONS) == 4
    assert {r["id"] for r in MODULE.RELATIONS} == {"cause", "effect", "temporal", "contrast"}


def test_solver_renders_complete_prose_with_joint_relation_equations():
    result = MODULE.run()
    assert result["stats"]["rendered"] == 12
    assert result["stats"]["exact"] == 0
    assert result["stats"]["admitted"] == 0
    assert all(row["independent_reparse"] for row in result["rendered_candidates"])
    assert all(row["audit"]["two_pointer_exact"] is False for row in result["rendered_candidates"])
    assert all(row["audit"]["sha_exact"] is False for row in result["rendered_candidates"])
    assert all(row["audit"]["anti_shortcut"]["word_order_mirror"] for row in result["rendered_candidates"])
    assert all(row["audit"]["anti_shortcut"]["catalogue_absent"] for row in result["rendered_candidates"])
    assert result["next_repair"].startswith("at the first residual character debt")


def test_independent_audit_detects_exact_tape_without_claiming_readability():
    row = MODULE.audit("A man a plan a canal Panama.")
    assert row["two_pointer_exact"] and row["sha_exact"]
    assert row["central_admission"]["local_catalogue_absent"] is False

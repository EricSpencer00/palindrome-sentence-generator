import importlib.util
from pathlib import Path
_spec = importlib.util.spec_from_file_location("encoder_lane", Path(__file__).parents[1] / "experiments/encoder_semantic_scene_pseudolikelihood_20260916.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
audits, complete_assignments, SCENES, novelty_preflight = _mod.audits, _mod.complete_assignments, _mod.SCENES, _mod.novelty_preflight

def test_complete_assignments_keep_semantic_choices_and_obligations():
    rows = complete_assignments(SCENES[0])
    assert len(rows) == 3
    assert all(r["complete"] and r["choices"] and isinstance(r["obligations"], list) for r in rows)

def test_independent_audit_reports_mismatch_and_hashes():
    result = audits("The careful gardener waters young roses beside an old wall.")
    assert result["letters"] >= 39
    assert result["exact"] is False
    assert result["two_pointer"] is False
    assert result["hash_equal"] is False

def test_registry_preflight_is_before_execution_and_unique():
    assert novelty_preflight()["passed"] is True

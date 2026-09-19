import importlib.util
from pathlib import Path

P = Path(__file__).parents[1] / "experiments/grammar_char_intersection_20260920.py"
spec = importlib.util.spec_from_file_location("gci", P); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

def test_orbit_rejects_mismatch_and_closes_matching_terminal():
    assert mod.orbit_pair("ab", "ba") == ""
    assert mod.orbit_pair("ab", "ca") is None

def test_run_has_complete_clauses_and_independent_audits():
    out = mod.run()
    assert not out["novelty_preflight"]["exact_signature_collision"]
    assert out["stats"]["complete_clauses"] == out["stats"]["rendered"]
    for row in out["rendered_candidates"]:
        assert row["semantic_closure"]["roles"]
        assert len(row["sha256"]) == 64
        assert row["two_pointer_pairs"] == row["letters"] // 2

def test_no_posthoc_repair_or_catalogue_import():
    out = mod.run()
    assert out["grammar"]["posthoc_repair"] is False
    assert out["novelty_preflight"]["catalogue_text_imported"] is False

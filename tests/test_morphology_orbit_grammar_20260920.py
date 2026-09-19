import importlib.util, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("lane", ROOT / "experiments/morphology_orbit_grammar_20260920.py")
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)

def test_morphology_precedes_lexicalization_and_orbit_audits():
    r = M.run(); assert r["novelty_preflight"]["status"] == "passed"
    assert r["novelty_preflight"]["signature_collision"] is False
    assert r["candidate_count"] == 16
    for row in r["rendered_candidates"]:
        assert row["orbit_obligation"]["lexicalized_before_emit"]
        assert row["independent_clause_authorship"]["distinct_templates"]
        assert row["audit"]["reverse_sha256_equal"] == row["audit"]["two_pointer_exact"]

def test_controls_are_complete_real_prose_and_shortcuts_banned():
    r = M.run()
    assert all(c["real_prose"] and c["audit"]["letters"] > 20 for c in r["real_prose_controls"])
    assert all(not any(row["anti_shortcut_flags"].values()) for row in r["rendered_candidates"])

def test_artifact_round_trip(tmp_path):
    data = M.run(); p = tmp_path / "artifact.json"; p.write_text(json.dumps(data)); loaded = json.loads(p.read_text())
    assert loaded["failure_and_repair"]["next_construction_discriminator"]

import importlib.util
from pathlib import Path

P = Path(__file__).parents[1] / "experiments/seed_seam_growth_20260917.py"
spec = importlib.util.spec_from_file_location("seed_seam_growth", P)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

def test_seed_control_is_exact_and_independently_audited():
    r = m.run()
    assert r["seed_control"]["audit"]["exact"]
    assert r["seed_control"]["audit"]["two_pointer_exact"]
    assert r["seed_control"]["audit"]["sha256"] == r["seed_control"]["audit"]["reverse_sha256"]

def test_frontier_has_rendered_provenance_and_no_shortcut_admission():
    r = m.run(); assert r["rows"]
    for row in r["rows"]:
        assert row["rendered"] and row["provenance"]["joint_character_zipper"]
        assert "first_mismatch" in row["audit"]
    assert not r["reader_eligible"]
    assert r["next_repair"]["operator"]

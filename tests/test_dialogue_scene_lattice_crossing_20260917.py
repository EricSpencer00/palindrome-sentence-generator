from experiments.dialogue_scene_lattice_crossing_20260917 import run

def test_dialogue_lattice_is_rendered_and_independently_rechecked():
    data = run()
    assert data["candidate_count"] == 3 * 4 * 4 * 3 * 3
    assert all(row["rendered"].endswith(".") for row in data["rendered_candidates"])
    assert all(row["center_crossing"]["word"] for row in data["rendered_candidates"])
    assert all(row["audit"]["two_pointer_exact"] == row["audit"]["exact"] for row in data["rendered_candidates"])
    assert data["novelty_preflight"]["status"] == "passed"
    assert data["reader_eligible"] is False

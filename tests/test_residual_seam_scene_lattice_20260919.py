from experiments.residual_seam_scene_lattice_20260919 import audit, run

def test_independent_audit():
    a=audit("An aide rips nine memos; some men inspire Diana.")
    assert a["two_pointer_exact"] and a["sha_equal"]

def test_indexed_lattice_has_complete_prose_states():
    r=run()
    assert r["stats"]["rendered"] > 0
    assert r["stats"]["indexed_states"] > 0
    assert all(x["provenance"]["finished_tape_reversed"] is False for x in r["actual_candidates"])
    assert all(x["residual_state"]["agreement"] for x in r["actual_candidates"])

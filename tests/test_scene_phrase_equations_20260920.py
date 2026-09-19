from experiments.scene_phrase_equations_20260920 import run

def test_scene_phrase_lattice_is_exact_and_audited():
    report = run()
    assert report["novelty_preflight"]["status"] == "passed"
    assert report["stats"]["exact"] == report["stats"]["candidates"]
    assert report["stats"]["controls"] == 3
    for row in report["candidates"]:
        assert row["exact_audit"]["two_pointer_exact"]
        assert all(eq["satisfied"] for eq in row["equations"])
        assert row["provenance"]["independently_authored_scene"]
        assert row["provenance"]["independently_authored_phrases"]
        assert not row["provenance"]["finished_tape_reversal"]
        assert not row["provenance"]["word_order_mirror"]
        assert not row["provenance"]["rlaif_candidate_scoring"]
        assert row["exact_audit"]["sha256_forward"] == row["exact_audit"]["sha256_reverse"]

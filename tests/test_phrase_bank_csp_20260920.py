from experiments.phrase_bank_csp_20260920 import run

def test_phrase_bank_is_broad_and_csp_audited():
    r=run()
    assert r["stats"]["nodes"] == 1080
    assert r["stats"]["exact"] == 0
    assert r["stats"]["mechanically_admitted"] == 0
    assert r["novelty_preflight"]["status"] == "passed"
    assert "simultaneous" in r["method"]
    assert any("SHA-256" in x for x in r["independent_audit"])

def test_controls_reject_fragments_and_no_reverse_provenance():
    r=run()
    assert all(not x["provenance"]["finished_tape_reversed"] for x in r["candidates"])
    assert all(not x["provenance"]["catalogue_imported"] for x in r["candidates"])

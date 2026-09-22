from experiments.authored_wordpair_obstruction_20260921 import run

def test_obstruction_is_bounded_and_ordinary():
    r = run(); assert r["status"] == "operator_abandoned_linguistic_obstruction"
    assert r["candidate_count"] == 2 and r["exact_count"] == 0
    for row in r["candidates"]:
        assert row["complete_prose"] and row["provenance"]["ordinary_english_only"]
        assert not row["provenance"]["reversed_token_surfaces"]

def test_independent_audits():
    for row in run()["candidates"]:
        a = row["audit"]; assert a["sha256_forward"] != a["sha256_reverse"]
        assert not a["two_pointer_exact"]

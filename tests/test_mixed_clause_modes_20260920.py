from experiments.mixed_clause_modes_20260920 import CLAUSES, online_audit, run

def test_all_requested_clause_modes_and_state_are_present():
    r = run()
    assert {c.mode for c in CLAUSES} == {"declarative", "imperative", "question", "copular", "dialogue"}
    assert r["config"]["agreement_attachment_tense_carried"]
    assert r["novelty_preflight"]["status"] == "passed"

def test_each_pair_has_independent_pointer_and_sha_audit():
    r = run()
    assert r["stats"]["pairs"] == len(r["rendered_candidates"])
    for row in r["rendered_candidates"]:
        assert len(row["audit"]["sha256_forward"]) == 64
        assert row["audit"]["sha256_reverse"] != row["audit"]["sha256_forward"] or not row["audit"]["two_pointer_exact"]
        assert all(not v for v in row["anti_shortcut_flags"].values())
    assert online_audit("ordinary prose")["two_pointer_exact"] is False

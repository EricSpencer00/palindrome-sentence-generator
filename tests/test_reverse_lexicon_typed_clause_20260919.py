from experiments.reverse_lexicon_typed_clause_20260919 import audit, run

def test_audit_independently_checks_tape():
    assert audit("An aide rips nine memos; some men inspire Diana.") ["exact"]
    assert not audit("this is prose") ["exact"]

def test_reverse_lexicon_is_joint_and_fail_closed():
    data = run(max_states=500)
    assert data["method"] == "reverse-lexicon-typed-clause-20260919"
    assert data["candidate_count"] == 0
    assert data["independent_audit"].startswith("sha256")

from experiments.discourse_stack_synchronous_20260920 import audit, consume, plans, prose_controls, run


def test_shared_plan_and_reader_controls():
    assert len(plans()) == 3
    assert {p.attachment for p in plans()} == {"pp", "adverb", "adjective"}
    assert len(prose_controls()) >= 20
    assert all(row["audit"] == audit(row["rendered"]) for row in prose_controls())


def test_live_residual_and_independent_audit():
    assert consume("abc", "cba") == ("", "")
    assert consume("abc", "ba") == ("c", "")
    assert consume("abc", "xyz") is None
    result = run(state_limit=20_000)
    assert result["stats"]["plans"] == 3
    assert result["provenance"]["explicit_open_constituent_stack"]
    for row in result["candidates"]:
        assert row["audit"] == audit(row["rendered"])
        assert row["audit"]["exact"]

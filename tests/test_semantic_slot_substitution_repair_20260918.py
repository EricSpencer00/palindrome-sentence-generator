from experiments.semantic_slot_substitution_repair_20260918 import (
    audit, clauses, residual, run, tape,
)


def test_typed_clause_bank_has_agreement_and_varied_slots():
    cs = clauses()
    assert len(cs) == 710
    assert len({c.text for c in cs}) > 100
    assert any(c.subject.number == "PL" for c in cs)
    assert any(c.adjunct.text == "at dawn" for c in cs)


def test_residual_is_computed_before_rendering():
    cs = clauses()
    left = next(c for c in cs if c.text == "an aide rips nine memos")
    right = next(c for c in cs if c.text == "some men inspire Diana")
    r = residual(left, right, ";")
    assert r["predicted_exact"]
    assert r["residual_mismatches"] == 0
    assert audit(left.text + ";" + right.text + ".")["two_pointer_exact"]


def test_bounded_run_has_independent_audit_and_no_new_admission():
    result = run(max_probes=250)
    assert result["stats"]["bounded"]
    assert result["stats"]["stored_probes"] == 250
    assert result["stats"]["new_mechanically_admitted"] == 0
    for row in result["exact_candidates"]:
        assert row["audit"]["two_pointer_exact"]
        assert row["audit"]["sha_equal_under_reversal"]

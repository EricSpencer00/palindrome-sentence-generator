from __future__ import annotations

from experiments.residual_typed_sentence_search_20260914 import _reparse, plans, run
from llm_palindrome.admission import normalize_letters


def test_residual_search_declares_free_midpoint_and_reader_gate() -> None:
    result = run(max_states=2_000)
    assert result["config"]["free_midpoint"] is True
    assert result["config"]["residual_driven_expansion"] is True
    assert result["config"]["machine_readability_certification"] is False
    assert result["reader_facing_next_test"]


def test_every_emitted_residual_closure_has_independent_audits() -> None:
    result = run(max_states=2_000)
    by_name = {plan.name: plan for plan in plans()}
    for row in result["records"]:
        rendered = row["rendered"]
        tape = normalize_letters(rendered)
        assert tape == tape[::-1]
        assert row["independent_normalized_letters"] == tape
        assert row["independent_reparse"]["ok"] is True
        assert _reparse(by_name[row["plan"]], tuple(row["words"]))["ok"] is True
        assert row["mechanically_eligible"] is all(row["mechanical_checks"].values())


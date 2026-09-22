import json
from pathlib import Path

from experiments.two_event_clause_product_nonterminal_20260930 import run
from llm_palindrome.validator import is_palindrome, normalize


def test_online_nonterminal_product_has_ordinary_residual_and_fail_closed_audit():
    data = run()
    assert data["novelty_preflight"]["passed"]
    assert data["search"]["online_character_matching"]
    assert data["summary"]["frontier_samples"] == 2
    for row in data["frontier_samples"]:
        assert row["letters"] > 38
        assert not row["two_pointer_exact"]
        assert not is_palindrome(row["rendered"])
        assert row["sha256_forward"] != row["sha256_reverse"]
        assert row["reader_gate"]["status"] == "closed"


def test_run_artifact_is_reproducible_and_marks_no_generated_exact():
    data = run()
    path = Path(__file__).parents[1] / "runs" / "two-event-clause-product-nonterminal-20260930.json"
    saved = json.loads(path.read_text())
    assert saved["experiment_id"] == data["experiment_id"]
    assert saved["summary"]["exact_novel"] == 0
    assert all(normalize(x["rendered"]) == x["normalized"] for x in saved["frontier_samples"])

from experiments.llm_authored_typed_bank_csp_20260919 import audit, run, search


def test_independent_audit_accepts_seed():
    result = audit("An aide rips nine memos; some men inspire Diana.")
    assert result["two_pointer_exact"] and result["sha_equal"]


def test_typed_bank_search_is_deterministic_and_not_reward_scored():
    result = run(40, 42)
    assert result["provenance"]["rlaif_per_candidate"] is False
    assert result["provenance"]["authoring_model"] == "gpt-oss:20b"
    assert result["novelty_preflight"]["status"] == "passed"
    frontier = search(78)["frontier_controls"]
    assert frontier
    assert all(row["reader_status"].startswith("frontier") for row in frontier)

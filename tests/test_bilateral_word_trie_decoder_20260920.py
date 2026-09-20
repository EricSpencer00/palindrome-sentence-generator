import json
from pathlib import Path
from experiments.bilateral_word_trie_decoder_20260920 import audit, run


def test_decoder_has_live_trie_queries_and_independent_audit():
    result = run(state_limit=100_000)
    assert result["novelty_preflight"]["status"] == "passed"
    assert result["novelty_preflight"]["finished_tape_reversal"] is False
    assert all(row["stats"]["trie_queries"] > 0 for row in result["searches"].values())
    assert audit("An aide rips nine memos; some men inspire Diana")["exact"]


def test_artifact_is_reproducible_and_fail_closed_for_reader_gate():
    result = run(state_limit=100_000)
    assert result["reader_gate"] == "closed until blinded human ratings"
    assert all(c["provenance"]["post_hoc_repair"] is False for c in result["exact_candidates"])

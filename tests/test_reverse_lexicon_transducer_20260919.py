from experiments.reverse_lexicon_transducer_20260919 import consume, run

def test_residual_invariant():
    assert consume("abc", "cba") == ("", "")
    assert consume("abcd", "cba") == ("d", "")
    assert consume("abc", "xyz") is None

def test_queried_transducer_is_fail_closed():
    data = run(limit=500)
    assert data["reverse_index_keys"] > 0
    assert data["candidate_count"] == 0
    assert data["invariant"].startswith("consume")

from experiments.two_clause_live_buffer_20260919 import SEED, audit, consume, search

def test_live_buffer_and_seed_regression():
    assert consume("abc", "cba") == ("", "")
    assert consume("abcd", "cba") == ("d", "")
    d=search()
    assert d["seed_regression"]["audit"]["exact"]
    assert d["stats"]["exact"] == len(d["exact_candidates"])
    assert all(not x["provenance"]["complete"] for x in d["candidates"])

from experiments.lexical_centerout_20260919 import compatible, exact, search

def test_boundary_compatibility_is_synchronous():
    assert compatible("a", "a")
    assert not compatible("the", "map")

def test_search_rejects_unfinished_nonprose_frontier_without_repair():
    states = search(2, 10)
    assert states == []

def test_exact_audit_independent_hashes():
    ok, hf, hr, n = exact("a man a plan")
    assert not ok and hf != hr and n == 9

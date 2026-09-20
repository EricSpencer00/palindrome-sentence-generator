from experiments.lexical_centerout_20260919 import compatible, exact, search

def test_boundary_compatibility_is_synchronous():
    assert compatible("a", "a")
    assert not compatible("the", "map")

def test_search_never_uses_posthoc_reversal_and_renders_text():
    states = search(2, 10)
    assert states
    for state in states:
        text = " ".join(state.left + state.right)
        assert text
        ok, hf, hr, n = exact(text)
        assert n > 0 and len(hf) == 64 and len(hr) == 64

def test_exact_audit_independent_hashes():
    ok, hf, hr, n = exact("a man a plan")
    assert not ok and hf != hr and n == 9

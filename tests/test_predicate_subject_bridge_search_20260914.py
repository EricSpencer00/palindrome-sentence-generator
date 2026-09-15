from experiments.predicate_subject_bridge_search_20260914 import (
    independent_audit,
    indexed_search,
)


def test_indexed_search_crosses_phrase_boundaries():
    left = (("ab",), ("c",))
    right = (("c",), ("ba",))
    rows, stats = indexed_search(left, right, state_budget=100)
    assert rows == [(('ab', 'c'), ('c', 'ba'))]
    assert stats["budget_exhausted"] is False
    assert independent_audit("ab c; c ba.")["exact"]


def test_independent_audit_rejects_mismatch():
    assert not independent_audit("The baker reads.")["exact"]

from experiments.multiclause_residual_bridge_20260914 import render_side
from experiments.predicate_subject_bridge_search_20260914 import (
    independent_audit,
    indexed_search,
)


def test_staggered_clause_boundaries_do_not_change_exact_tape():
    left = (("ab",), ("c",), ("d",))
    right = (("d",), ("cb",), ("a",))
    rows, _ = indexed_search(left, right, state_budget=100)
    assert rows
    left_text = render_side(rows[0][0], (2, 3))
    right_text = render_side(rows[0][1], (1, 3))
    assert independent_audit(left_text + " " + right_text)["exact"]


def test_render_side_requires_complete_declared_clauses():
    assert render_side(("an aide", "reads", "a note"), (3,)) == "An aide reads a note."

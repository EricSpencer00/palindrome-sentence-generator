import pytest
from experiments import lexicalized_constituent_interior_csp_20260920 as solver


@pytest.mark.parametrize("right,accepted", [("cba", True), ("ccba", True), ("dcba", False)])
def test_constituent_solver_keeps_terminal_middle_inside_word(monkeypatch, right, accepted):
    monkeypatch.setattr(solver, "LEXICON", {"TOKEN": ("ab", right)})
    monkeypatch.setattr(solver, "GRAMMAR", {"S_SG": (("TOKEN",),)})
    report = solver.search(max_states=100, min_letters=1)
    rows = [row for row in report["exact_candidates"] if row["rendered"] == f"ab; {right}."]
    assert bool(rows) == accepted
    assert all(row["audit"]["exact"] and row["audit"]["sha_equal"] for row in report["exact_candidates"])


def test_equal_remaining_words_do_not_count_as_a_nonempty_center(monkeypatch):
    monkeypatch.setattr(solver, "LEXICON", {"TOKEN": ("ab", "ba")})
    monkeypatch.setattr(solver, "GRAMMAR", {"S_SG": (("TOKEN",),)})
    report = solver.search(max_states=100, min_letters=1)
    assert report["exact_candidates"]
    assert report["stats"]["nonempty_center_closures"] == 0
    assert all(row["center_residual"] == "" for row in report["exact_candidates"])

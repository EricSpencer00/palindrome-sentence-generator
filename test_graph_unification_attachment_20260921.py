from graph_unification_attachment_20260921 import online_support

def test_incremental_frontier_survives_first_terminal():
    ok, checks, frontier = online_support("ab", "ba")
    assert len(frontier) > 1
    assert checks >= 1


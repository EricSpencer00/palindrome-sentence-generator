from experiments.boundary_residual_typed_trie_20260914 import build_trie, run
from experiments.syntax_first_clause_pair_20260914 import enumerate_clauses


def test_trie_indexes_complete_reversed_tapes():
    rows = enumerate_clauses()[:20]
    trie = build_trie(rows)
    assert len(trie) > 1
    assert all(isinstance(node.children, dict) for node in trie)


def test_bounded_run_reports_joint_residual_evidence():
    result = run(max_left=1000, max_pairs=3)
    assert result["config"]["complete_typed_clause_on_both_sides"]
    assert result["config"]["character_residual_pruning"]
    assert result["inventory"]["traversed_left_clauses"] == 1000
    assert result["exact_closure_count_seen"] == len(result["rendered_candidates"])
    assert result["scope"].startswith("Bounded joint")

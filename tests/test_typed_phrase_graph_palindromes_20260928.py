from experiments.typed_phrase_graph_palindromes_20260928 import (
    LEFT, RIGHT, exact_two_pointer, pair_graph, run, validator,
)


def test_graph_closes_with_independent_audits():
    result = run()
    candidate = result["candidate"]
    assert candidate["letters"] == 156
    assert candidate["exact_two_pointer"] is True
    assert candidate["validator"] is True
    assert candidate["sha256"] == candidate["independent_forward_reverse_sha256"]
    assert candidate["novelty_preflight"] is True
    assert candidate["provenance"]["self_palindromic_units"] is False
    assert candidate["provenance"]["posthoc_character_repair"] is False


def test_frontier_has_typed_edges_not_a_finished_tape_lookup():
    edges = pair_graph()
    assert len(edges) == len(LEFT)
    assert {e["left"].kind for e in edges} == {"vocative-perception"}
    assert {e["right"].kind for e in edges} == {"copular-return"}
    assert all(e["frontier"]["matched"] == len(e["left"].tape) for e in edges)

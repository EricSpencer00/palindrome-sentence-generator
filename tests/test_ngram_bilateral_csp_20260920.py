from experiments.ngram_bilateral_csp_20260920 import load_observed_edges, ngram_bilateral_csp
from experiments.forward_lexicalized_grammar_20260920 import ATOMIC


def test_observed_lattice_contains_real_edges():
    edges, rows = load_observed_edges(limit=100)
    assert rows == 100
    assert ("one", "of") in edges


def test_ngram_bilateral_reports_bounded_solver_state():
    edges, _ = load_observed_edges(limit=100)
    result = ngram_bilateral_csp(ATOMIC, edges, max_nodes=50)
    assert result["stats"]["status"] in {"SAT", "UNSAT", "timeout"}
    assert result["provenance"]["candidate_reranking"] is False

from wordpath_ngram_csp_20260920 import load_wordpath_lattice, wordpath_csp


def test_wordpath_lattice_loads_observed_edges():
    vocab, nexts, prevs, rows = load_wordpath_lattice(edge_limit=1000, vocab_limit=200)
    assert rows == 1000
    assert vocab and nexts and prevs


def test_wordpath_search_is_bounded_and_not_a_reranker():
    vocab, nexts, prevs, _ = load_wordpath_lattice(edge_limit=100, vocab_limit=40)
    result = wordpath_csp(vocab, nexts, prevs, max_words=6, max_nodes=30)
    assert result["stats"]["status"] in {"SAT", "UNSAT", "timeout"}
    assert result["provenance"]["candidate_reranking"] is False

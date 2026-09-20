from experiments.vocative_bilateral_grammar_20260920 import has_nested_word_span, run


def test_nested_span_filter_rejects_anchor_wrapping():
    assert has_nested_word_span("Leon an aide rips nine memos some men inspire Diana Noel".split())


def test_vocative_lane_is_bounded_and_filters_nested_spans():
    result = run(max_nodes=200)
    assert result["stats"]["status"] in {"SAT", "UNSAT", "timeout"}
    assert result["provenance"]["nested_palindrome_spans_rejected"]

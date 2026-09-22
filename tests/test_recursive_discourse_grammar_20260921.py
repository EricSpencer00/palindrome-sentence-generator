from experiments.recursive_discourse_grammar_20260921 import run


def test_independent_frame_loop_is_not_reported_as_recursive_composition():
    result = run(max_depth=2, max_states=100)
    assert result["status"] == "rejected_as_independent_frame_loop"
    assert result["provenance"]["recursive_loop"] is False
    assert result["provenance"]["composed_across_depths"] is False
    assert result["provenance"]["independent_frame_queries"] is True

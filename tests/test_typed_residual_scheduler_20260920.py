from experiments.typed_residual_scheduler_20260920 import _nested_span, run


def test_nested_gate_rejects_a_wrapped_inner_palindrome():
    assert _nested_span(("leon", "deer", "deliver", "drawer", "reward", "reviled", "reed", "noel"))


def test_typed_scheduler_is_bounded():
    result = run(lexicon_limit=15, max_words=10, max_nodes=50000, beam_width=5000)
    assert result["stats"]["status"] in {"SAT", "UNSAT", "timeout"}
    assert result["provenance"]["one_sided_residual_scheduler"]

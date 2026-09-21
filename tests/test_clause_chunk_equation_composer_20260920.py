from experiments.clause_chunk_equation_composer_20260920 import (
    LEFT, RIGHT, audit, run, segment, stream_equation,
)


def test_stream_checks_independent_chunks_online():
    assert stream_equation(("ab", "c"), ("c", "ba"))["accepted"]
    failed = stream_equation(("ab", "x"), ("c", "ba"))
    assert not failed["accepted"]
    assert failed["first_mismatch"]["offset"] == 2


def test_run_records_controls_provenance_and_no_exact_shortcut():
    result = run()
    expected = sum((len(left.split()) - 3) * (len(right.split()) - 3)
                   for left in LEFT for right in RIGHT)
    assert result["stats"]["segment_states"] == expected
    assert result["novelty_preflight"]["status"] == "passed"
    assert result["novelty_preflight"]["mirrored_chunk_pairs"] is False
    assert result["controls"]
    assert result["next_repair"]["operator"] == "typed residual continuation bank"

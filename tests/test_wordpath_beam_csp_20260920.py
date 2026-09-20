from experiments.wordpath_beam_csp_20260920 import _admissible, run


def test_admission_rejects_repeated_or_mirrored_words():
    assert not _admissible(("level", "stone"))
    assert not _admissible(("red", "blue", "blue", "red"))


def test_small_beam_run_is_bounded():
    result = run(edge_limit=30000, vocab_limit=100, max_words=8, beam_width=1000)
    assert result["stats"]["status"] in {"SAT", "UNSAT_OR_BEAM_EXHAUSTED"}
    assert result["provenance"]["one_sided_residual_advancement"]

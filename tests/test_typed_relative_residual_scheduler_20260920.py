from experiments.typed_relative_residual_scheduler_20260920 import GRAMMAR, run


def test_relative_productions_are_present():
    assert ("DET", "N", "REL") in GRAMMAR["OBJ"]
    assert ("RELPRON", "V", "DET", "N") in GRAMMAR["REL"]


def test_relative_scheduler_is_bounded():
    result = run(lexicon_limit=15, max_words=12, max_nodes=50000, beam_width=5000)
    assert result["stats"]["status"] in {"SAT", "UNSAT", "timeout"}
    assert result["provenance"]["nested_span_rejected"]

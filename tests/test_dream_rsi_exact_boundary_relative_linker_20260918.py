from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1]))

from experiments.dream_rsi_exact_boundary_relative_linker_20260918 import run


def test_relative_linker_is_the_next_distinct_repair():
    payload = run()
    assert payload["construction"]["heldout_right_frame"] == [
        "det", "subject", "linker", "verb", "name"
    ]
    assert payload["stats"]["fresh_nodes"] > 0
    assert payload["stats"]["fresh_exact"] == 0
    assert payload["novelty_preflight"]["duplicate_sweep"] is False
    assert payload["reader_gate"]["status"] == "not_triggered"
    assert all(row["provenance"]["heldout_relative_linker_frame"] for row in payload["rendered_candidates"])

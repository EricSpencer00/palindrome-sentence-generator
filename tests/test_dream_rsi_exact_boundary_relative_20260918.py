from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1]))

from experiments.dream_rsi_exact_boundary_relative_20260918 import run


def test_heldout_auxiliary_frame_is_a_distinct_bounded_repair():
    payload = run()
    assert payload["construction"]["heldout_right_frame"] == [
        "det", "subject", "aux", "verb", "name"
    ]
    assert payload["stats"]["fresh_nodes"] > 0
    assert payload["stats"]["fresh_exact"] == 0
    assert payload["reader_gate"]["status"] == "not_triggered"
    assert payload["novelty_preflight"]["duplicate_sweep"] is False
    for row in payload["rendered_candidates"]:
        assert row["audit"]["sha256_forward"] != row["audit"]["sha256_reverse"]
        assert row["provenance"]["heldout_right_frame"]

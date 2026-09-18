import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from experiments.three_region_discourse_anchor_20260917 import run, letters


def test_bounded_three_region_lane_has_intact_prose_and_independent_audits():
    payload = run()
    assert payload["stats"]["rendered"] == 16
    assert payload["stats"]["exact"] == 0
    assert all(row["rendered"].endswith(".") for row in payload["rendered_candidates"])
    for row in payload["rendered_candidates"]:
        tape = letters(row["rendered"])
        assert row["audit"]["sha256_forward"]
        assert row["audit"]["letters"] == len(tape)
        assert len(row["regions"]) == 3
        assert row["provenance"]["catalogue_used"] is False

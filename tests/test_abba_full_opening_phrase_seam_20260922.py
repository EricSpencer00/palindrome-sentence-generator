import importlib.util
from pathlib import Path

P = Path(__file__).parents[1] / "experiments/abba_full_opening_phrase_seam_20260922.py"
spec = importlib.util.spec_from_file_location("lane", P)
lane = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lane)


def test_full_opening_lane_records_authored_controls_and_no_false_closure():
    data = lane.run()
    assert data["stats"]["pairs"] == 4
    assert data["stats"]["rendered_candidates"] == 4
    assert data["stats"]["exact_gt38"] == 0
    assert data["stats"]["fully_consumed_openings"] == 2
    assert all(row["provenance"]["complete_authored_prose"] for row in data["rendered_candidates"])


def test_full_opening_residuals_are_mechanically_audited():
    data = lane.run()
    for row, cert in zip(data["rendered_candidates"], data["residual_certificates"]):
        assert row["audit"]["two_pointer_exact"] is False
        assert cert["reverse_obligation_prefix"]
        assert "residual" in cert

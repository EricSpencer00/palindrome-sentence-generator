import importlib.util
from pathlib import Path

P = Path(__file__).parents[1] / "experiments/abba_function_word_seam_20260922.py"
spec = importlib.util.spec_from_file_location("lane", P)
lane = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lane)

def test_function_word_lane_has_rendered_controls_and_residuals():
    data = lane.run()
    assert data["stats"]["opening_classes"] == 4
    assert data["stats"]["rendered_candidates"] == 16
    assert data["stats"]["exact_gt38"] == 0
    assert all(row["provenance"]["complete_authored_prose"] for row in data["rendered_candidates"])
    assert all("residual" in row for row in data["rendered_candidates"])

def test_function_word_reverse_onsets_are_mechanically_checked():
    data = lane.run()
    for opening in ("some", "an", "no", "a"):
        rows = [r for r in data["residual_certificates"] if r["opening"] == opening]
        assert any(r["expected_reverse_onset"] == opening and
                   r["observed_opening"] == opening for r in rows)

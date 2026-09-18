import importlib.util
from pathlib import Path

PATH = Path(__file__).parents[1] / "experiments/center_out_role_zipper_20260918.py"
spec = importlib.util.spec_from_file_location("center_out", PATH)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

def test_center_out_is_bounded_and_independently_audited():
    result = mod.run()
    assert result["stats"]["rendered"] == 8
    assert result["stats"]["beam_width"] == 24
    assert result["stats"]["exact"] == 0
    for row in result["rendered_candidates"]:
        assert row["provenance"]["catalogue_used"] is False
        assert row["provenance"]["finished_tape_reversal"] is False
        assert row["audit"]["sha256_forward"] != row["audit"]["sha256_reverse"]
        assert row["audit"]["two_pointer_exact"] is False


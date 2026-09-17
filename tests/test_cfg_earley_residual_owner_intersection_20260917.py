import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("cfg_owner_lane", ROOT / "experiments/cfg_earley_residual_owner_intersection_20260917.py")
lane = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = lane
spec.loader.exec_module(lane)


def test_cfg_owner_lane_has_complete_prose_and_independent_audits():
    result = lane.run()
    assert result["novelty_preflight"]["passed"]
    assert len(result["rows"]) == 3
    for row in result["rows"]:
        assert row["letters"] > 100
        assert row["left_chart"]["accepted"] and row["right_chart"]["accepted"]
        assert row["coordination_chart"]["accepted"]
        assert row["exact_check_direct_reverse"]["exact"] is False
        assert row["exact_check_opposing_index"]["exact"] is False
        assert row["independent_exact_agreement"]
        assert row["online_intersection"]["residual_owner_invariant"]
        assert row["anti_shortcut_flags"]["catalogue_text"] is False

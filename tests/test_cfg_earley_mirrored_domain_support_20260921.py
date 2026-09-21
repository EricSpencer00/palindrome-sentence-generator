import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("cfg_domain_lane", ROOT / "experiments/cfg_earley_mirrored_domain_support_20260921.py")
lane = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = lane
spec.loader.exec_module(lane)


def test_earley_domain_lane_records_conflicts_and_independent_audits():
    result = lane.run()
    assert result["novelty_preflight"]["passed"]
    assert len(result["rows"]) == 3
    for item in result["rows"]:
        assert item["letters"] >= 39
        assert item["left_chart"]["accepted"] and item["right_chart"]["accepted"]
        assert item["exact_check_pointer"]["exact"] is False
        assert item["exact_check_sha"]["exact"] is False
        assert item["independent_exact_agreement"]
        assert item["mirrored_character_domains"]["conflict_count"] > 0
        assert item["anti_shortcut_flags"]["fixed_tape"] is False

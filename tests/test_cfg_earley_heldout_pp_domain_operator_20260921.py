import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("cfg_heldout_pp", ROOT / "experiments/cfg_earley_heldout_pp_domain_operator_20260921.py")
lane = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = lane
spec.loader.exec_module(lane)


def test_heldout_pp_is_pre_render_and_independently_audited():
    result = lane.run()
    assert result["novelty_preflight"]["passed"]
    assert len(result["rows"]) == 3
    for row in result["rows"]:
        assert row["letters"] >= 39
        assert row["left_chart"]["accepted"] and row["right_chart"]["accepted"]
        assert row["provenance"]["heldout_pp_bank"]
        assert row["exact_check_pointer"]["exact"] is False
        assert row["exact_check_sha"]["exact"] is False
        assert row["independent_exact_agreement"]
        assert row["anti_shortcut_flags"]["post_render_repair"] is False

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("lane", ROOT / "experiments/cfg_earley_character_equation_forest_20260916.py")
lane = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = lane
spec.loader.exec_module(lane)


def test_cfg_earley_character_equation_forest_is_fresh_and_audited():
    result = lane.run()
    assert result["novelty_preflight"]["passed"]
    assert len(result["rows"]) == 2
    assert all(row["letters"] > 100 for row in result["rows"])
    assert all(row["earley_chart_left"]["accepted"] for row in result["rows"])
    assert all(row["earley_chart_right"]["accepted"] for row in result["rows"])
    assert all(row["rendered"].endswith(".") and ". " in row["rendered"] for row in result["rows"])
    assert all(not row["exact_check_two_pointer"]["exact"] for row in result["rows"])
    assert all(row["exact_check_two_pointer"]["exact"] == row["exact_check_sha256"]["exact"] for row in result["rows"])
    assert all(not row["anti_shortcut_flags"]["fixed_tape"] for row in result["rows"])
    assert "PP-production" in result["next_repair"]

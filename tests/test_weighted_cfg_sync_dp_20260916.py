import json
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_weighted_cfg_run_is_frozen_and_has_no_reader_candidate():
    run = json.loads((ROOT / "runs/weighted-cfg-sync-dp-20260916.json").read_text())
    assert run["signature"].startswith("weighted-cfg-synchronous-parse-forest|")
    assert run["base"]["left_derivations"] == 648
    assert run["base"]["right_derivations"] == 648
    assert run["repair"]["left_derivations"] == 3240
    assert run["repair"]["right_derivations"] == 3240
    assert run["base"]["exact_count"] == 0
    assert run["repair"]["exact_count"] == 0
    assert run["reader_eligible"] == []

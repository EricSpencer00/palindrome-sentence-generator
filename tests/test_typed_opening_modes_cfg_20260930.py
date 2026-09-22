import json
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/typed_opening_modes_cfg_20260930.py"
ARTIFACT = ROOT / "runs/typed-opening-modes-cfg-20260930.json"


def test_cfg_lane_writes_audited_artifact():
    runpy.run_path(str(SCRIPT), run_name="__main__")
    data = json.loads(ARTIFACT.read_text())
    assert data["method"] == "typed_opening_modes_cfg_live_character_intersection"
    assert data["search"]["attempts"] > 0
    assert data["control"]["audit"]["exact_two_pointer"]
    assert data["control"]["generated"] is False
    for row in data["generated_exact"]:
        assert row["audit"]["exact_two_pointer"]
        assert row["audit"]["sha256_forward"] == row["audit"]["sha256_reverse"]


def test_no_reversed_phrase_bank_or_posthoc_repair():
    source = SCRIPT.read_text()
    assert "reversed phrase" in source
    assert "post-hoc repair" in source

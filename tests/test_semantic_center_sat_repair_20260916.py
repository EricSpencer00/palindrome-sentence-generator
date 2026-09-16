import json
from pathlib import Path

def test_repair_is_debt_guided_and_center_frozen():
    x=json.loads(Path("runs/semantic-center-sat-repair-20260916.json").read_text())
    assert x["stats"]["exact"] == 0
    assert x["repaired"]["center_event"] == x["seed"]["center_event"]
    assert x["repaired"]["sat_character_equation"]["residual_debt"] <= x["seed"]["sat_character_equation"]["residual_debt"]
    assert x["repair_trace"][0]["residual_after"] < x["repair_trace"][0]["residual_before"]
    assert all(not c["provenance"]["reversed_finished_sentence"] for c in (x["seed"],x["repaired"],x["heldout"]))

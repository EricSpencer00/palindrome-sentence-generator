import json
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "runs/typed-phrase-graph-noel-war-window-edit-20260930.json"


def test_noel_war_window_is_exact_and_immutable():
    runpy.run_path(str(ROOT / "experiments/typed_phrase_graph_noel_war_window_edit_20260930.py"), run_name="__main__")
    data = json.loads(ART.read_text())
    row = data["candidate"]
    assert row["left"] == "Noel, did I draw?"
    assert row["right"] == "Ward, I did, Leon."
    assert row["window_tape_reverse"]
    assert row["audit"]["two_pointer_exact"]
    assert row["audit"]["validator_exact"]
    assert row["audit"]["sha_equal"]
    assert row["audit"]["letters"] == 240
    assert row["outside_tape_preserved"]
    assert data["provenance"]["window_diff_only"]

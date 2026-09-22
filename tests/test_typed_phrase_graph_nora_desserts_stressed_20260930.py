import json
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "runs/typed-phrase-graph-nora-desserts-stressed-20260930.json"


def test_joint_nora_desserts_window_is_exact_and_immutable():
    runpy.run_path(
        str(ROOT / "experiments/typed_phrase_graph_nora_desserts_stressed_20260930.py"),
        run_name="__main__",
    )
    data = json.loads(ART.read_text())
    row = data["candidate"]
    assert row["left"] == "Nora, was I stressed?"
    assert row["right"] == "Desserts, I saw, Aron."
    assert row["window_tape_reverse"]
    assert row["outside_tape_preserved"]
    assert row["audit"]["two_pointer_exact"]
    assert row["audit"]["validator_exact"]
    assert row["audit"]["sha_equal"]
    assert row["audit"]["letters"] == 240
    assert data["provenance"]["window_diff_only"]

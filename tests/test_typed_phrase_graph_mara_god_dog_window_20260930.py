import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.typed_phrase_graph_mara_god_dog_window_20260930 import audit


ROOT = Path(__file__).resolve().parents[1]


def test_selected_mara_dog_window_is_exact_and_independent():
    data = json.loads((ROOT / "runs/typed-phrase-graph-mara-god-dog-window-20260930.json").read_text())
    selected = data["selected"]
    assert selected["window"] == {"left": "Mara, was I God?", "right": "Dog, I saw, Aram."}
    assert selected["outside_tape_preserved"]
    assert selected["audit"]["two_pointer_exact"]
    assert selected["audit"]["validator_exact"]
    assert selected["audit"]["sha_equal"]
    assert audit(selected["candidate"]) == selected["audit"]

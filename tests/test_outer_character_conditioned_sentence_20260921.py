import json
from pathlib import Path

def test_outer_classes_are_equal_and_match_rendered_tape():
    data = json.loads(Path("runs/outer-character-conditioned-sentence-20260921.json").read_text())
    assert data["stats"]["retained_outer_assignments"] > 0
    for row in data["diagnostic_controls"]:
        raw = "".join(ch for ch in row["rendered"].lower() if ch.isalpha())
        choice = row["outer_choice"]
        assert choice["start_class"] == choice["end_class"]
        assert raw[0] == choice["start_class"]
        assert raw[-1] == choice["end_class"]

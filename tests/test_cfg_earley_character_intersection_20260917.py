import json
from pathlib import Path

RUN = Path("runs/cfg-earley-character-intersection-20260917.json")

def test_cfg_intersection_artifact_is_independently_audited():
    data = json.loads(RUN.read_text())
    assert data["method"].startswith("seedless bounded CFG chart")
    assert data["candidate_count"] == 3060
    assert data["exact_count"] == 0
    assert data["candidates"]
    for row in data["candidates"]:
        a = row["audit"]
        assert a["two_pointer"] == (a["hash_reverse_equal"])
        assert row["anti_shortcut"] == {
            "word_order_only": False, "repeated_unit": False,
            "catalogue_source": False, "fragment": False}

def test_next_repair_is_concrete_and_not_a_duplicate_sweep():
    text = json.loads(RUN.read_text())["next_repair"]
    assert "relative-clause" in text
    assert "held-out inflection" in text

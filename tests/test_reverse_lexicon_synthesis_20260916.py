import json
import re
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_reverse_lexicon_keeps_complete_prose_and_zero_closure():
    run = json.loads((ROOT / "runs/reverse-lexicon-synthesis-20260916.json").read_text())
    assert run["stats"] == {"candidates": 4, "exact": 0, "admitted": 0}
    assert len(run["candidates"]) == 4
    for row in run["candidates"]:
        tape = "".join(re.findall(r"[A-Za-z]", row["rendered"])).lower()
        assert len(tape) == row["letters"] >= 39
        assert row["exact_audit"]["independent_pointer_audit"] is False
        assert row["exact_audit"]["hash_equal"] is False
        assert row["admitted"] is False
        assert row["provenance"]["source_sentences_copied"] is False
        assert row["provenance"]["reversed_finished_sentence"] is False


def test_reverse_lexicon_preserves_a_long_intact_probe():
    run = json.loads((ROOT / "runs/reverse-lexicon-synthesis-20260916.json").read_text())
    assert max(row["letters"] for row in run["candidates"]) == 81
    assert all(row["checks"]["complete_prose"] for row in run["candidates"])

import json
from pathlib import Path


def test_slot_pair_search_is_cross_word_and_exact_by_construction():
    data = json.loads(Path("runs/slot-pair-character-search-20260919.json").read_text())
    assert data["stats"]["exact"] == 0
    assert data["stats"]["pruned"] == 36
    assert data["candidates"] == []
    assert data["provenance"] == {
        "templates": [["det", "adj", "subject", "verb", "det", "object"], ["det", "subject", "verb", "det", "object", "adjunct"]],
        "finished_tape_reversal": False,
        "paired_clauses": False,
        "aligned_token_mirror": False,
        "fallback": False,
        "distinct_words": True,
    }

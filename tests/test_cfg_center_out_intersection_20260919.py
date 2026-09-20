import json
from pathlib import Path


RUN = Path("runs/cfg-center-out-intersection-20260919.json")


def test_cfg_center_out_prunes_before_completion_without_shortcuts():
    data = json.loads(RUN.read_text())
    assert data["stats"] == {
        "derivations": 0,
        "states": 0,
        "pruned": 72,
        "exact": 0,
        "longest_letters": 0,
    }
    assert data["provenance"]["single_sentence_derivation"]
    assert data["provenance"]["distinct_slots"]
    assert not data["provenance"]["paired_clauses"]
    assert not data["provenance"]["aligned_token_pairs"]
    assert data["candidates"] == []

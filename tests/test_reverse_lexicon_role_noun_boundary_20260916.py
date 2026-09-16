import json
import re
from pathlib import Path


def test_role_noun_boundary_repair_is_complete_and_unadmitted():
    path = Path("runs/reverse-lexicon-role-noun-boundary-20260916.json")
    run = json.loads(path.read_text())
    assert run["stats"] == {"candidates": 1, "exact": 0, "admitted": 0}
    row = run["candidates"][0]
    tape = "".join(re.findall(r"[A-Za-z]", row["rendered"])).lower()
    assert row["letters"] == len(tape) == 93
    assert row["exact_audit"]["exact"] is False
    assert row["exact_audit"]["hash_equal"] is False
    assert row["admitted"] is False
    assert row["checks"]["complete_prose"] is True
    assert row["provenance"]["source_sentences_copied"] is False

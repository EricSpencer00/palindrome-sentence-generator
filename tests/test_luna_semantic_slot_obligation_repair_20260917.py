import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).parents[1]


def tape(s):
    return re.sub(r"[^a-z]", "", s.casefold())


def test_typed_slot_repair_preserves_complete_prose_and_independent_audits():
    run = json.loads((ROOT / "runs/luna-semantic-slot-obligation-repair-20260917.json").read_text())
    assert run["novelty_preflight"]["passed"]
    assert run["stats"] == {"seeds": 2, "controls": 2, "repair_frontier": 64, "rendered_total": 66, "exact": 0, "mechanically_admitted": 0, "longest_letters": 131}
    assert len(run["controls"]) == 2
    for row in run["controls"] + run["rows"]:
        normalized = tape(row["rendered"])
        audit = row["exact_audit"]
        assert normalized and audit["letters"] == len(normalized)
        assert audit["sha_forward"] == hashlib.sha256(normalized.encode()).hexdigest()
        assert audit["sha_reverse"] == hashlib.sha256(normalized[::-1].encode()).hexdigest()
        assert audit["two_pointer_exact"] == audit["sha_exact"]
        assert row["semantic_witness"]["complete_ordinary_prose"]
        assert row["semantic_witness"]["roles_preserved"]
        assert row["semantic_witness"]["agreement_preserved"]
        assert row["provenance"]["seed_was_near_miss_not_catalogue"]
        assert not row["provenance"]["catalogue_imported"]
        assert not row["provenance"]["reversed_finished_sentence"]
        assert not row["provenance"]["word_order_mirror"]
        assert row["next_repair"]
    assert all(r["phase"] == "seed_control" for r in run["controls"])
    assert all(r["phase"] == "typed_slot_repair" for r in run["rows"])

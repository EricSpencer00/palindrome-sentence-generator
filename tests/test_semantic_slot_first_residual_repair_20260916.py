import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).parents[1]


def tape(text):
    return "".join(re.findall(r"[A-Za-z]", text)).lower()


def test_fresh_prose_slot_repairs_have_independent_exact_audits_and_no_shortcuts():
    run = json.loads((ROOT / "runs/semantic-slot-first-residual-repair-20260916.json").read_text())
    assert run["novelty_preflight"]["passed"] is True
    assert run["stats"] == {"fresh_seeds": 3, "rendered": 6, "repairs": 3, "exact": 0, "mechanically_admitted": 0, "max_letters": 108}
    for row in run["rows"]:
        normalized = tape(row["rendered"])
        audit = row["exact_audit"]
        assert normalized and audit["letters"] == len(normalized)
        assert audit["sha256_forward"] == hashlib.sha256(normalized.encode()).hexdigest()
        assert audit["sha256_reverse"] == hashlib.sha256(normalized[::-1].encode()).hexdigest()
        assert row["rendered"].endswith(".")
        assert row["provenance"]["catalogue_imported"] is False
        assert row["provenance"]["reversed_finished_sentence"] is False
        assert row["provenance"]["word_order_mirror"] is False
        assert row["semantic_witness"]["complete_ordinary_clauses"] is True
    repairs = [r for r in run["rows"] if r["phase"] == "repair"]
    assert all(r["residual_before"] for r in repairs)
    assert all(r["semantic_witness"]["slot_only_edit"] for r in repairs)
    assert run["next_repair"]

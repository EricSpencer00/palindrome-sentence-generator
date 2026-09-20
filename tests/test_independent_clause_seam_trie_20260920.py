import json
from pathlib import Path

from experiments.independent_clause_seam_trie_20260920 import audit, norm


def test_audit_uses_independent_forward_reverse_digests():
    row = audit("Artists watch quiet harbors at dawn. Pilots guide old bridges after rain.", {}, {})
    tape = norm(row["rendered"])
    assert row["forward_sha256"] != row["reverse_sha256"]
    assert row["independent_two_pointer"] is False
    assert row["letters"] == len(tape)


def test_run_records_no_borrowed_or_mirrored_shortcut():
    run = json.loads(Path("runs/independent-clause-seam-trie-20260920.json").read_text())
    assert run["provenance"]["independent_derivations"]
    assert run["novelty_preflight"]["finished_tape_reversal"] is False
    assert all(not r["anti_shortcut"]["source_sentence"] for r in run["candidates"])

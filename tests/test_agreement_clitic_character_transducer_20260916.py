from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from tools.audit_candidate_readability_20260915 import audit as aggregate_audit

ROOT = Path(__file__).parents[1]


def tape(text: str) -> str:
    return "".join(re.findall(r"[A-Za-z]", text)).lower()


def test_agreement_clitic_transducer_emits_one_scene_and_closes_local_obligations():
    run = json.loads((ROOT / "runs/agreement-clitic-character-transducer-20260916.json").read_text())
    assert run["novelty_preflight"]["status"] == "passed"
    assert run["novelty_preflight"]["duplicate_sweep"] is False
    assert run["semantic_scene"]["text"].endswith(".")
    assert run["semantic_scene"]["roles"] == ["agent", "event", "instrument", "recipient", "temporal-setting"]
    assert run["lexicalization"]["all_obligations_solved"] is True
    assert all(row["matched"] for row in run["lexicalization"]["obligations"])
    assert run["lexicalization"]["features"]["subject_number"] == "singular"
    assert run["lexicalization"]["features"]["tense"] == "present"
    audit = run["audit"]
    normalized = tape(run["semantic_scene"]["text"])
    assert audit["normalized_tape"] == normalized
    assert audit["exact"] is False
    assert audit["independent_two_pointer_exact"] is False
    assert audit["sha256_forward"] == hashlib.sha256(normalized.encode()).hexdigest()
    assert audit["sha256_reverse"] == hashlib.sha256(normalized[::-1].encode()).hexdigest()
    assert audit["sha256_forward"] != audit["sha256_reverse"]
    assert all(value is False for value in run["anti_shortcut_flags"].values())
    assert run["provenance"]["catalogue_lookup"] is False
    assert run["next_repair"]["operator"]


def test_shared_audit_surfaces_semantic_scene_text():
    report = aggregate_audit([ROOT / "runs/agreement-clitic-character-transducer-20260916.json"])
    assert report["candidate_count"] == 1
    assert report["exact_count"] == 0
    expected = "The patient pilot checks the engine, notes its gauge, and tells the crew it starts at dawn."
    assert report["rows"][0]["rendered"] == expected

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).parents[1]


def letters(text: str) -> str:
    return "".join(re.findall(r"[A-Za-z]", text)).lower()


def test_targeted_word_internal_repair_is_novel_role_preserving_and_independently_rejected():
    run = json.loads((ROOT / "runs/word-internal-seam-first-mismatch-repair-20260916.json").read_text())
    assert run["novelty_preflight"]["status"] == "passed"
    assert run["novelty_preflight"]["duplicate_sweep"] is False
    assert run["repair"]["changed_word_count"] == 2
    assert run["repair"]["preserved_roles"] == ["agent", "event", "patient", "setting"]
    assert run["repair"]["preserved_internal_seam"] == "teacher/recorder: e == e"
    assert run["rendered_intact_scene"].endswith(".")
    audit = run["audit"]
    tape = letters(run["rendered_intact_scene"])
    assert audit["independent_ascii_tape"] == tape
    assert audit["independent_two_pointer_exact"] is False
    assert audit["exact"] is False
    assert audit["sha256_forward"] == hashlib.sha256(tape.encode()).hexdigest()
    assert audit["sha256_reverse"] == hashlib.sha256(tape[::-1].encode()).hexdigest()
    assert audit["sha256_forward"] != audit["sha256_reverse"]
    assert all(value is False for value in run["anti_shortcut_flags"].values())
    assert run["provenance"]["catalogue_lookup"] is False
    assert run["next_repair"]

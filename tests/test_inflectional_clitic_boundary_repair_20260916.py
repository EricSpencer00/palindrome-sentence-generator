from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).parents[1]
RUN = ROOT / "runs/inflectional-clitic-boundary-repair-20260916.json"


def tape(text: str) -> str:
    return "".join(re.findall(r"[A-Za-z]", text)).lower()


def test_fresh_boundary_dp_has_prose_repair_and_independent_exact_audit():
    run = json.loads(RUN.read_text())
    assert run["novelty_preflight"]["status"] == "passed"
    assert run["novelty_preflight"]["duplicate_sweep"] is False
    assert run["stats"] == {"base_rendered": 5, "repair_rendered": 5, "flat_rendered": 10, "exact_repairs": 0}
    assert len(run["rendered_rows"]) == 10
    assert all(row["rendered"].endswith(".") for row in run["rendered_rows"])
    for row in run["candidates"]:
        base = row["base"]["rendered"]
        repaired = row["repair"]["rendered"]
        assert len(tape(base)) >= 50 and len(tape(repaired)) >= 50
        assert base.endswith(".") and repaired.endswith(".")
        assert row["repair"]["heldout"] is True
        assert row["repair"]["agreement_preserved"] is True
        audit = row["repair"]["audit"]
        normalized = tape(repaired)
        assert audit["normalized_tape"] == normalized
        assert audit["exact"] is False
        assert audit["independent_two_pointer_exact"] is False
        assert audit["sha256_forward"] == hashlib.sha256(normalized.encode()).hexdigest()
        assert audit["sha256_reverse"] == hashlib.sha256(normalized[::-1].encode()).hexdigest()
        assert all(value is False for value in run["anti_shortcut_flags"].values())
        assert row["reader_gate"]["intact_prose"] is True
        assert row["reader_gate"]["human_readability_certified"] is False
        assert row["provenance"]["catalogue_lookup"] is False


def test_boundary_dp_carries_agreement_and_clitic_state():
    run = json.loads(RUN.read_text())
    for row in run["candidates"]:
        dp = row["base"]["boundary_dp"]
        assert dp["features"]["number"] == "singular"
        assert dp["features"]["tense"] == "present"
        assert dp["features"]["clitic"] == "the"
        assert dp["states_explored"] >= 8
        assert len(dp["seams"]) == dp["states_explored"]

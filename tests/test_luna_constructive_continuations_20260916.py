"""Contract tests for the three post-ten-lane Luna continuations."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).parents[1]


def tape(text: str) -> str:
    return "".join(re.findall(r"[A-Za-z]", text)).lower()


def independent(text: str) -> tuple[str, bool, bool]:
    value = tape(text)
    i, j = 0, len(value) - 1
    exact = bool(value)
    while i < j:
        if value[i] != value[j]:
            exact = False
            break
        i += 1
        j -= 1
    forward = hashlib.sha256(value.encode()).hexdigest()
    reverse = hashlib.sha256(value[::-1].encode()).hexdigest()
    return value, exact, forward == reverse


def load(name: str) -> dict:
    return json.loads((ROOT / "runs" / name).read_text())


def test_paired_lexical_lane_has_fresh_prose_and_dual_audit() -> None:
    run = load("paired-lexical-grammar-20260916.json")
    assert run["pairs_examined"] == 6
    assert run["exact_count"] == 0
    assert run["novelty_preflight"]["collisions"] == []
    for row in run["candidates"]:
        value, pointer, hashed = independent(row["text"])
        assert value and len(value) >= 70
        assert pointer is False and hashed is False
        assert row["provenance"]["fresh_semantic_frames"]
        assert row["provenance"]["catalogue_imported"] is False


def test_free_center_bridge_records_prose_provenance_and_repair() -> None:
    run = load("free-center-semantic-bridge-20260916-luna.json")
    value, pointer, hashed = independent(run["rendered"])
    assert len(value) == run["independent_exact_audit"]["letters"] == 109
    assert pointer is False and hashed is False
    assert run["novelty_preflight"]["passed"] is True
    assert run["provenance"]["catalogue_text_used"] is False
    assert run["next_repair_operator"]["operator"] == "first-residual-free-bridge-swap"


def test_productive_grammar_growth_is_not_a_duplicate_sweep() -> None:
    run = load("grammar-pair-composition-20260916.json")
    assert run["status"] == "diagnostic_no_candidate"
    assert run["growth_lengths"] == [55, 114, 168, 223]
    assert run["base"]["exact_count"] == 0
    assert run["repair"]["exact_count"] == 0
    assert run["novelty_preflight"]["catalogue_match"] is False
    assert run["provenance"]["repeated_content_units"] is False
    assert run["repair"]["repair_operator"]
    for row in run["base"]["candidates"] + run["repair"]["candidates"]:
        value, pointer, hashed = independent(row["rendered"])
        assert value and pointer is False and hashed is False
        assert row["complete_prose"] is True
        assert row["word_order_mirror"] is False


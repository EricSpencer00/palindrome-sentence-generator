"""Invariants for the character-synchronous grammar product."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "earley_finite_state_grammar_intersection_20260916",
    ROOT / "experiments/earley_finite_state_grammar_intersection_20260916.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_registry_preflight_is_fresh_and_artifact_is_not_colliding():
    result = MODULE.novelty_preflight()
    assert result["status"] == "passed"
    assert not result["signature_overlaps"]
    assert not result["artifact_collisions"]


def test_agreement_is_rejected_during_lexical_transition():
    frame = MODULE.FRAMES["event"]
    sem = MODULE.Semantics(subject_number="sg")
    slot = frame.slots[2]
    assert all(item.number == "sg" for item in MODULE._choices(slot, sem))
    assert MODULE._apply(slot, MODULE.Lexeme("guide", "verb", "pl", "guide"), sem) is None


def test_independent_audit_does_not_trust_search_counter():
    text = "A quiet artist reads a letter near the garden; the garden near a letter reads artist quiet a."
    audit = MODULE.independent_audit(text)
    assert audit["normalized_tape"] == audit["independent_ascii_tape"]
    assert audit["exact"] == audit["independent_two_pointer_exact"]
    assert audit["letters"] > MODULE.MIN_LETTERS


def test_lockstep_product_keeps_semantics_in_closure_rows():
    rows, dead, stats = MODULE.search_pair(MODULE.FRAMES["event"], MODULE.FRAMES["event"], max_states=20_000)
    assert stats["states"] > 0
    for row in rows:
        assert {role.split(":", 1)[0] for role in row["left_semantics"]["roles"]} >= {"agent", "event", "patient", "setting"}
        assert {role.split(":", 1)[0] for role in row["right_semantics"]["roles"]} >= {"agent", "event", "patient", "setting"}
        assert row["letters_consumed_lockstep"] == row["audit"]["letters"]


def test_run_records_long_rendered_probes_and_next_repair():
    result = MODULE.run(max_states=2_000)
    probes = result["rendered_probes"]
    assert probes
    assert max(row["audit"]["letters"] for row in probes) > 38
    assert result["independent_audit"] == [row["audit"] for row in result["rendered_candidates"]]
    assert "deepest dead Earley state" in result["next_repair_operator"]

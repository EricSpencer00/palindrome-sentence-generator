from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "weighted_morphology_fst_lockstep_20260916",
    ROOT / "experiments/weighted_morphology_fst_lockstep_20260916.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_registry_preflight_passes_for_new_signature():
    result = MODULE.novelty_preflight()
    assert result["status"] == "passed"
    assert not result["signature_overlaps"]


def test_morphology_register_rejects_number_mismatch_during_transition():
    slot = MODULE.GRAMMARS["event"].slots[2]
    state = MODULE.Morphology(subject_number="sg")
    plural = MODULE.Lexeme("carry", "verb", "pl", "pres", "carry")
    assert MODULE.apply(slot, plural, state) is None


def test_morphology_register_can_be_solved_from_reverse_order():
    # The reverse side encounters setting noun before its determiner.  The
    # finite-state register stores that number and checks the later determiner.
    noun_slot = MODULE.GRAMMARS["event"].slots[7]
    det_slot = MODULE.GRAMMARS["event"].slots[6]
    state = MODULE.apply(noun_slot, MODULE.Lexeme("arena", "noun", "sg"), MODULE.Morphology())
    assert state is not None and state.locative_number == "sg"
    assert MODULE.apply(det_slot, MODULE.Lexeme("some", "det", "pl"), state) is None


def test_audit_hash_and_two_pointer_are_independent():
    row = MODULE.audit("A nerd carries a candle near the garden.")
    assert row["normalized_tape"] == row["independent_ascii_tape"]
    assert row["independent_sha256"]
    assert row["exact"] == row["independent_two_pointer_exact"]


def test_run_optimizes_and_renders_long_probe_with_repair_operator():
    result = MODULE.run(max_states=2_000)
    assert result["rendered_probes"]
    assert max(row["audit"]["letters"] for row in result["rendered_probes"]) > 38
    assert "held-out inflectional variant" in result["next_repair_operator"]
    assert result["stats"]["weighted_transitions"] >= result["stats"]["character_pairs_emitted"]
    repair = result["mismatch_directed_morphology_repair"]
    assert len(repair["held_out_variants"]) == 4
    assert "dead_frontier" in repair

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "typed_cfg_exact_tape_resegmentation_20260916",
    ROOT / "experiments/typed_cfg_exact_tape_resegmentation_20260916.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_fresh_constructor_is_exact_and_long():
    source = MODULE.construct_exact_tape()
    assert source["letters"] > 38
    assert source["tape"] == source["tape"][::-1]
    assert source["left_chunks"] != source["right_chunks"]


def test_registry_preflight_records_near_pair_without_signature_collision():
    result = MODULE.novelty_preflight()
    assert result["status"] == "passed"
    assert result["conceptual_near_pairs"]
    assert not result["signature_overlaps"]


def test_cfg_valency_transition_rejects_intransitive_patient():
    state = MODULE.CFGState("object", subject_number="sg", predicate="sleep", valency="intransitive")
    noun = MODULE.TypedWord("candle", "noun", "sg")
    assert MODULE.transition(state, noun) is None


def test_dp_preserves_tape_and_records_typed_partial_probe():
    source = MODULE.construct_exact_tape()
    dictionary, typed = MODULE.typed_lexicon()
    rows, partial, stats = MODULE.segment(source["tape"], dictionary, typed)
    assert stats["typed_edges_accepted"] > 0
    assert partial
    assert partial[0]["cfg_state"]["phase"] in {"object", "after_object"}
    assert not rows


def test_run_has_independent_source_audit_and_repair_without_tape_mutation():
    result = MODULE.run()
    source = result["source_exact_tape"]
    assert source["independent_exact"]
    assert source["mechanical_audit"]["source_tape_unchanged"]
    assert result["rendered_candidates_and_probes"]
    assert result["mismatch_directed_repair"]["status"] == "executed"
    assert result["mismatch_directed_repair"]["tape_mutation"] is False

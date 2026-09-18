"""Tests for progressive residual expansion and pre-search island loss."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.whole_text_palindrome_product_20260913 import compile_slots, construct, exhaustive_reference
from experiments.progressive_phrase_residual_product_20260913 import (
    DERIVATIONS, Derivation, Frame, QUIET, VISITED, EAGER, PERSONS, endpoint_algebra,
    independent_parse, progressive_construct, render, replay_pair_ledger, run,
    lex,
)


def test_endpoint_index_requires_three_pairs_and_residual_ledger_is_real():
    algebra = endpoint_algebra()
    assert algebra["minimum_matched_pairs"] == 3
    assert algebra["joins"]
    assert all(row["matched_pairs"] >= 3 for row in algebra["joins"])
    result = run(max_states=1_000)
    assert result["eligible_derivation_count"] == len(result["derivation_runs"])
    assert all(item["kernel"]["ledger_count"] >= item["kernel"]["deepest_matched_pairs"] + 1 for item in result["derivation_runs"])


def test_online_span_loss_prunes_actual_inner_palindrome_after_outer_pair():
    frame = Frame("synthetic", "svo_typed", PERSONS[0], EAGER, VISITED,
                  lex("arena", "noun", "place"), QUIET)
    grammar = compile_slots((("q",), ("ab",), ("c",), ("ba",), ("q",)))
    derivation = Derivation(frame, ("left", "inner_a", "inner_b", "inner_c", "right"),
                            tuple((lex(word, "token", "function"),) for word in ("q", "ab", "c", "ba", "q")))
    result = progressive_construct(grammar, derivation, max_states=100)
    assert result["online_span_pruned"] >= 1
    assert result["span_prune_events"][0]["prune"]["words"] == ["ab", "c", "ba"]
    assert result["records"] == []


def test_independent_parse_and_kernel_oracle_remain_separate():
    assert all(independent_parse(d.identifier, render(tuple(slot[0].form for slot in d.choices)))["ok"] for d in DERIVATIONS)
    slots = (("ij", "ix"), ("k",), ("ji", "zz"))
    result = construct(compile_slots(slots), max_states=1_000)
    assert {tuple(row["words"]) for row in result["records"]} == exhaustive_reference(slots) == {("ij", "k", "ji")}


def test_pair_ledger_replay_is_independent_of_generator_state():
    assert replay_pair_ledger("ij k ix.", {"pair_trace": [[1, "i"], [2, "j"]]})["ok"] is False
    assert replay_pair_ledger("ij k ji.", {"pair_trace": [[1, "i"], [2, "j"]]})["ok"] is True


def test_fresh_run_reports_no_readability_claim_and_exact_rows_are_replayed():
    result = run(max_states=1_000)
    assert result["config"]["progressive_residual_expansion"]
    assert result["reader_facing_next_test"].startswith("Only an admitted closure")
    for row in result["exact_closures"]:
        assert row["independent_terminal_path_replay"]["ok"]
        assert row["independent_parse"]["ok"]
        assert row["independent_exact_audit"]["exact"]

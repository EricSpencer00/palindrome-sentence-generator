import hashlib
import json
from pathlib import Path

from experiments.global_synchronous_dependency_chart_20260922 import (
    CharacterTrie,
    ChartState,
    Lexeme,
    PLANS,
    ParseState,
    advance_parse,
    boundary_positions,
    initial_parse,
    parse_complete,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments" / "global_synchronous_dependency_chart_20260922.py"
ARTIFACT = ROOT / "runs" / "global-synchronous-dependency-chart-20260922.json"


def payload():
    return json.loads(ARTIFACT.read_text())


def lexeme(surface, slot, *, lemma=None, number="", tense="", frames=(), domains=()):
    return Lexeme(
        surface=surface,
        lemma=lemma or surface,
        slot=slot,
        number=number,
        tense=tense,
        person="third" if slot in {"subject", "verb"} else "",
        valency="transitive" if slot == "verb" else "",
        semantic_types=domains,
        brown_count=10,
        brown_source="synthetic-test",
        wordnet_frames=frames,
    )


def test_remote_artifact_is_source_identical_and_sol_bench_bounded():
    row = payload()
    assert row["provenance"]["host"] == "hst-bench"
    assert row["provenance"]["python"] == "3.12.3"
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == row["provenance"]["source_sha256"]
    assert row["fixed_domain"]["max_packed_states"] == 500_000
    assert row["fixed_domain"]["per_slot_frequency_cap"] == 64
    assert row["fixed_domain"]["lexical_widening_after_run"] is False
    assert row["search"]["cap_reached"] is False
    assert row["search"]["domain_exhausted"] is True


def test_preflight_distinguishes_global_chart_from_retired_local_stack():
    row = payload()
    preflight = row["novelty_preflight"]
    assert preflight["status"] == "passed"
    assert preflight["registry_read"] is True
    assert preflight["bounded_comparisons"] == preflight["registry_entries_checked"]
    assert preflight["exact_signature_collisions"] == []
    assert "global dual finite-clause dependency state" in preflight["distinction"]
    assert row["fixed_domain"]["retired_local_residual_stack"] is False


def test_state_contract_carries_both_parses_character_debt_and_masks():
    row = payload()
    contract = row["state_contract"]
    assert set(contract["parse_state"]) >= {
        "open_valencies",
        "subject_head",
        "predicate_head",
        "object_head",
        "agreement_resolved",
        "tense",
        "discourse_referents",
    }
    assert contract["character_state"] == ["owner", "residual", "left_cursor", "right_cursor"]
    assert contract["boundary_state"] == ["left_boundary_mask", "right_boundary_mask"]
    assert "completed_phrase" in contract["packing_key_excludes"]
    assert contract["terminal_access"] == "slot-specific forward/reverse character tries"
    # The concrete immutable state type also contains every promised field.
    fields = ChartState.__dataclass_fields__
    assert set(fields) >= {
        "left", "right", "owner", "residual", "left_cursor", "right_cursor",
        "left_boundary_mask", "right_boundary_mask", "used_content_lemmas",
    }


def test_character_trie_returns_only_prefix_comparable_terminals_in_each_direction():
    rows = (
        lexeme("area", "object", domains=("noun.location",)),
        lexeme("army", "object", domains=("noun.group",)),
        lexeme("art", "object", domains=("noun.artifact",)),
    )
    forward = CharacterTrie(rows, reverse=False)
    assert {row.surface for row in forward.compatible("ar")} == {"area", "army", "art"}
    assert {row.surface for row in forward.compatible("area") } == {"area"}
    backward = CharacterTrie(rows, reverse=True)
    assert {row.surface for row in backward.compatible("a")} == {"area"}
    assert backward.compatible("z") == ()


def test_dependency_automaton_closes_heads_valencies_agreement_and_tense():
    plan = PLANS[0]
    state = initial_parse(plan, reverse=False)
    transitions = (
        lexeme("the", "determiner"),
        lexeme("pilots", "subject", lemma="pilot", number="plural", domains=("noun.person",)),
        lexeme("guide", "verb", number="plural", tense="present", frames=(8,)),
        lexeme("a", "determiner"),
        lexeme("boat", "object", number="singular", domains=("noun.artifact",)),
    )
    for terminal in transitions:
        state = advance_parse(state, terminal, reverse=False)
        assert state is not None
    assert parse_complete(state, reverse=False)
    assert state.open_valencies == ()
    assert state.agreement_resolved is True
    assert state.valency_resolved is True
    assert state.subject_head == "pilot"
    assert state.predicate_head == "guide"
    assert state.object_head == "boat"
    assert state.tense == "present"
    assert {ref[0] for ref in state.discourse_referents} == {"subject", "object"}


def test_brown_wordnet_domain_is_common_capped_and_has_no_names_or_phrases():
    row = payload()
    inventory = row["inventory"]
    assert inventory["terminals"] == {"determiner": 2, "subject": 64, "object": 64, "verb": 64}
    assert inventory["minimum_brown_count"] == 4
    assert inventory["noun_lemmas"] > 100_000
    assert inventory["verb_lemmas"] > 10_000
    fixed = row["fixed_domain"]
    assert fixed["proper_names"] is False
    assert fixed["fragments"] is False
    assert fixed["completed_phrase_banks"] is False
    assert fixed["catalogue_text"] is False
    assert fixed["repeated_units"] is False
    assert fixed["seed_recovery"] is False
    assert fixed["post_hoc_repair"] is False


def test_exhausted_chart_preserves_precise_deepest_obstruction_and_promotes_nothing():
    row = payload()
    search = row["search"]
    assert search["stats"] == {
        "states_expanded": 10,
        "trie_queries": 10,
        "trie_terminals_returned": 10,
        "terminal_attempts": 10,
        "packed_cells_created": 6,
        "packed_cells": 10,
        "dead_cells": 4,
        "complete_cells": 0,
        "audited_candidates": 0,
        "accepted_candidates": 0,
    }
    assert search["audited_candidates"] == []
    assert search["accepted_candidates"] == []
    obstruction = search["obstruction"]
    assert obstruction["domain_exhausted"] is True
    assert obstruction["matched_cursor"] == 1
    assert obstruction["state"]["owner"] == "right"
    assert obstruction["state"]["residual"] == "era"
    assert obstruction["witness"] == {"left_tokens": ["a"], "right_tokens_reverse": ["area"]}
    assert obstruction["next_required_side"] == "left"
    assert obstruction["next_required_slot"] == "subject"
    probe = obstruction["best_terminal_probe"]
    assert probe["terminal"]["surface"] == "end"
    assert probe["matching_prefix_characters"] == 1
    assert probe["expected_next_character"] == "r"
    assert probe["observed_next_character"] == "n"
    assert row["verdict"] == "precise exhausted packed-chart obstruction"


def test_boundary_masks_round_trip_as_cursor_positions():
    mask = (1 << 1) | (1 << 4) | (1 << 9)
    assert boundary_positions(mask) == [1, 4, 9]

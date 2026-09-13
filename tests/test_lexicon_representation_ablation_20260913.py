"""Staged tests for the matched lexicon × representation ablation."""
from itertools import product

from experiments.lexicon_representation_ablation_20260913 import (
    HARD_WORD_COUNTS, build_cells, chart_kernel, run, semantic_review_payload,
)


def test_tiny_oracle_matches_the_exact_character_product():
    slots = (("ab", "zz"), ("cd", "xy"), ("cba",))
    result = chart_kernel(slots, grammar_id="oracle", seed=7, state_budget=1000)
    actual = {entry["tape_id"] for entry in result["closures"]}
    expected = {__import__("hashlib").sha256("".join(words).encode()).hexdigest()
                for words in product(*slots) if "".join(words) == "".join(words)[::-1]}
    assert result["exhausted"]
    assert actual == expected


def test_factor_and_lexicon_are_frozen_before_endpoint_expansion():
    cells = build_cells()
    assert set(cells) == {"H0", "H1", "S0", "S1"}
    assert cells["H0"]["lexicon"] == cells["S0"]["lexicon"]
    assert cells["H1"]["lexicon"] == cells["S1"]["lexicon"]
    assert {row["word_count"] for row in cells["H0"]["grammars"]} == set(HARD_WORD_COUNTS)
    assert {row["word_count"] for row in cells["S1"]["grammars"]} == set(HARD_WORD_COUNTS)
    assert "selection_sha256" in cells["H1"]["lexicon_evidence"]
    assert "before grammar construction" in cells["H1"]["lexicon_evidence"]["selection_timing"]
    # S1 is the complete independently selected L1 source; H1's 16-word
    # POS choice is an explicit, fixed hard-layout balancing decision.
    h1_words = {word for row in cells["H1"]["grammars"] for slot in row["slots"] for word in slot}
    s1_words = {word for row in cells["S1"]["grammars"] for slot in row["slots"] for word in slot}
    assert len(s1_words) > len(h1_words) * 10
    assert s1_words - h1_words


def test_identity_accounting_and_blind_payload_contract():
    result = run(expansions_per_cell=1000, seed=11)
    for cell in result["cells"].values():
        accounting = cell["closure_identity_accounting"]
        assert accounting["closures"] >= accounting["unique_tapes"]
        assert accounting["all_identity_mappings_verified"]
        for closure in cell["closure_audit"]:
            payload = closure["semantic_review"]
            assert set(payload) == {"review_id", "surface", "instruction"}
            assert all(word not in str(payload).lower() for word in ("layout", "template", "h0", "h1", "s0", "s1"))
            assert closure["promotion"] == "forbidden; diagnostic closure only"


def test_1000_expansion_smoke_is_deterministic_and_audited():
    left = run(expansions_per_cell=1000, seed=19)
    right = run(expansions_per_cell=1000, seed=19)
    assert left == right
    for cell in left["cells"].values():
        assert cell["unique_canonical_state_expansions"] <= 1000
        assert set(cell["mutually_exclusive_terminations"]).isdisjoint({"semantic_ranked"})
        assert isinstance(cell["coaccessible_next_letters"], dict)
    assert len(left["cells"]["S1"]["eligible_endpoint_vocabularies"]["start"]) > len(left["cells"]["H1"]["eligible_endpoint_vocabularies"]["start"])
    assert set(left["cells"]["S1"]["eligible_endpoint_vocabularies"]["start"]) - set(left["cells"]["H1"]["eligible_endpoint_vocabularies"]["start"])

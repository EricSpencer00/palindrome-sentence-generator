"""Contract checks for the three newest orthogonal Luna construction lanes."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).parents[1]


def tape(text: str) -> str:
    return "".join(re.findall(r"[A-Za-z]", text)).lower()


def test_discourse_relation_lane_has_complete_prose_and_two_independent_audits():
    run = json.loads((ROOT / "runs/discourse-relation-involution-20260916.json").read_text())
    assert "distinct families" in run["preflight"]["overlap_classification"]
    assert len(run["candidates"]) == 12
    assert run["exact_count"] == 0
    assert run["independent_audit"]["two_pointer_checked"] == 12
    assert run["independent_audit"]["sha_checked"] == 12
    assert all(row["rendered"].endswith(".") for row in run["candidates"])
    assert all(len(tape(row["rendered"])) >= 40 for row in run["candidates"])
    assert run["repair_at_first_residual"]["operator"]


def test_constrained_edit_program_preserves_intact_scene_and_monotone_debt():
    run = json.loads((ROOT / "runs/constrained-edit-program-constructor-20260916.json").read_text())
    assert run["novelty_preflight"]["passed"] is True
    assert len(run["states"]) == 4
    debts = [state["audit"]["mirrored_character_debt"] for state in run["states"]]
    assert debts == sorted(debts, reverse=True)
    assert all(state["text"].endswith(".") for state in run["states"])
    assert all(state["parse_meaning_contract"]["passed"] for state in run["states"])
    assert all(not state["audit"]["two_pointer_exact"] for state in run["states"])
    assert all(
        state["audit"]["sha_forward_reverse_exact"] is False
        and state["audit"]["independent_exact_agreement"] is True
        for state in run["states"]
    )


def test_append_algebra_emits_only_complete_clauses_and_records_invariant_failure():
    run = json.loads((ROOT / "runs/arbitrary-clause-macro-algebra-20260916.json").read_text())
    assert run["novelty_preflight"]["passed"] is True
    assert len(run["states"]) == 4
    assert run["exact_count"] == 0
    assert run["reader_eligible"] is False
    for state in run["states"]:
        rendered = state["rendered"]
        normalized = tape(rendered)
        assert rendered[0].isupper()
        assert rendered.endswith(".")
        assert normalized
        assert state["after"]["two_pointer_exact"] is False
        assert state["after"]["sha256_forward"] == hashlib.sha256(normalized.encode()).hexdigest()
        assert state["after"]["sha256_reverse"] == hashlib.sha256(normalized[::-1].encode()).hexdigest()
    assert run["repair_operator"]


def test_aggregate_surfaces_all_three_new_lane_routes():
    report = json.loads((ROOT / "runs/parallel-luna-readability-diagnostics-20260916.json").read_text())
    assert report["candidate_count"] == 5529
    assert report["exact_count"] == 84
    by_source = {row["source_run"]: row for row in report["route_summary"]}
    assert by_source["runs/constrained-edit-program-constructor-20260916.json"]["rows"] == 4
    assert by_source["runs/arbitrary-clause-macro-algebra-20260916.json"]["rows"] == 4
    assert by_source["runs/discourse-relation-involution-20260916.json"]["rows"] == 12
    assert by_source["runs/endpoint-aware-bilateral-seam-20260916.json"]["rows"] == 6
    assert by_source["runs/fresh-scene-tape-cfg-resegmentation-20260916.json"]["rows"] == 2
    assert by_source["runs/exact-candidate-slot-repair-neighborhood-20260916.json"]["rows"] == 6
    assert by_source["runs/compositional-slot-boundary-dp-20260916.json"]["rows"] == 6
    assert by_source["runs/semantic-phrase-edge-graph-joiner-20260916.json"]["rows"] == 3
    assert by_source["runs/coupled-object-attachment-repair-20260916.json"]["rows"] == 4
    assert by_source["runs/semordnilap-role-clause-product-20260916.json"]["rows"] == 4
    assert by_source["runs/typed-reversible-clause-composer-20260916.json"]["rows"] == 2
    assert by_source["runs/seed-extension-frame-insertion-20260916.json"]["rows"] == 6
    assert by_source["runs/interrogative-relative-template-solver-20260916.json"]["rows"] == 2
    assert by_source["runs/live-tape-clause-terminal-decoder-20260916.json"]["rows"] == 6
    assert by_source["runs/bilateral-semantic-growth-grammar-20260916.json"]["rows"] == 3
    assert by_source["runs/scene-lattice-attachment-csp-20260916.json"]["rows"] == 2
    assert by_source["runs/boundary-shift-semordnilap-grammar-20260916.json"]["rows"] == 2
    assert by_source["runs/boundary-shift-scene-equation-lattice-20260916.json"]["rows"] == 2
    assert by_source["runs/semantic-boundary-macro-fsm-20260916.json"]["rows"] == 3
    assert by_source["runs/past-tense-dependency-transducer-20260916.json"]["rows"] == 6
    assert by_source["runs/reverse-tape-relative-resegment-20260916.json"]["rows"] == 2
    assert by_source["runs/scene-semordnilap-graph-20260916.json"]["rows"] == 2
    assert by_source["runs/centerout-open-word-seam-20260916.json"]["rows"] == 2
    assert by_source["runs/brown-phrase-pair-seam-20260916.json"]["rows"] == 6
    assert by_source["runs/bilateral-role-lattice-repair-20260916.json"]["rows"] == 4
    assert by_source["runs/feature-carrying-center-cfg-20260916.json"]["rows"] == 2
    assert by_source["runs/cross-boundary-phrase-block-grammar-20260916.json"]["rows"] == 3
    assert by_source["runs/typed-boundary-block-scene-20260916.json"]["rows"] == 16
    assert by_source["runs/cross-pos-semordnilap-scene-cfg-20260916.json"]["rows"] == 2
    assert by_source["runs/outside-in-heldout-scene-20260916.json"]["rows"] == 3
    assert by_source["runs/connected-scene-joint-resegment-20260916.json"]["rows"] == 4
    assert by_source["runs/valency-clitic-live-lexicalizer-20260916.json"]["rows"] == 2
    assert by_source["runs/scalable-outsidein-phrase-pair-20260916.json"]["rows"] == 3
    assert by_source["runs/fresh-seed-benchmark-seam-growth-20260916.json"]["rows"] == 16
    assert by_source["runs/reversible-phrase-pair-scene-search-20260916.json"]["rows"] == 2
    assert by_source["runs/scalable-outsidein-paired-terminal-repair-20260916.json"]["rows"] == 3
    assert by_source["runs/fresh-seam-heldout-joint-repair-20260916.json"]["rows"] == 1
    assert by_source["runs/reversible-phrase-pair-role-repair-20260916.json"]["rows"] == 2
    assert by_source["runs/scalable-outsidein-opposing-terminal-repair2-20260916.json"]["rows"] == 3
    assert by_source["runs/fresh-seam-attachment-followup-20260916.json"]["rows"] == 1
    assert by_source["runs/reversible-phrase-directional-adjunct-repair-20260916.json"]["rows"] == 2
    assert by_source["runs/fresh-seam-connector-followup-20260916.json"]["rows"] == 1
    assert by_source["runs/scalable-outsidein-single-edge-repair3-20260916.json"]["rows"] == 1
    assert by_source["runs/reversible-phrase-determiner-repair-20260916.json"]["rows"] == 1
    assert by_source["runs/live-cfg-character-chart-20260916.json"]["rows"] == 6
    assert by_source["runs/fresh-crossword-seam-csp-20260916.json"]["rows"] == 16
    assert by_source["runs/corpus-seam-fresh-scene-grammar-20260916.json"]["rows"] == 2
    assert by_source["runs/live-cfg-chart-terminal-edge-repair-20260916.json"]["rows"] == 1
    assert by_source["runs/fresh-crossword-seam-csp-followup-20260916.json"]["rows"] == 1
    assert by_source["runs/corpus-seam-reauthored-repair-20260916.json"]["rows"] == 1
    assert by_source["runs/bidirectional-phrase-pair-growth-20260916.json"]["rows"] == 2
    assert by_source["runs/centerout-museum-scene-lattice-20260916.json"]["rows"] == 4
    assert by_source["runs/morphology-crossword-transducer-20260916.json"]["rows"] == 2
    assert by_source["runs/bidirectional-phrase-pair-adjunct-repair-20260916.json"]["rows"] == 1
    assert by_source["runs/centerout-museum-heldout-repair-20260916.json"]["rows"] == 1
    assert by_source["runs/morphology-crossword-single-repair-20260916.json"]["rows"] == 1
    assert by_source["runs/lexical-mirror-hypergraph-clause-20260916.json"]["rows"] == 1
    assert by_source["runs/dialogue-relative-clause-csp-20260916.json"]["rows"] == 8
    assert by_source["runs/function-word-boundary-dp-20260916.json"]["rows"] == 2
    assert by_source["runs/lexical-mirror-hypergraph-edge-repair-20260916.json"]["rows"] == 1
    assert by_source["runs/dialogue-relative-clause-followup-20260916.json"]["rows"] == 1
    assert by_source["runs/function-word-boundary-single-repair-20260916.json"]["rows"] == 1
    assert by_source["runs/lexical-mirror-hypergraph-verb-edge-repair-20260916.json"]["rows"] == 1
    assert by_source["runs/dialogue-speaker-followup-20260916.json"]["rows"] == 1
    assert by_source["runs/function-word-auxiliary-single-repair-20260916.json"]["rows"] == 1
    assert by_source["runs/active-passive-attachment-csp-20260916.json"]["rows"] == 8
    assert by_source["runs/semantic-slot-cross-boundary-dp-20260916.json"]["rows"] == 2
    assert by_source["runs/bespoke-scene-lattice-free-center-20260916.json"]["rows"] == 8
    assert by_source["runs/active-passive-attachment-followup-20260916.json"]["rows"] == 1
    assert by_source["runs/bespoke-scene-lattice-single-slot-repair-20260916.json"]["rows"] == 1
    assert by_source["runs/semantic-slot-single-object-repair-20260916.json"]["rows"] == 1
    assert by_source["runs/finite-reverse-phrase-composition-bank-20260916.json"]["rows"] == 4
    assert by_source["runs/centerfree-clause-pair-csp-20260916.json"]["rows"] == 4
    assert by_source["runs/finite-semantic-palindrome-csp-20260916.json"]["rows"] == 2
    assert by_source["runs/finite-reverse-phrase-single-boundary-repair-20260916.json"]["rows"] == 1
    assert by_source["runs/centerfree-clause-pair-followup-20260916.json"]["rows"] == 1
    assert by_source["runs/finite-semantic-csp-object-verb-repair-20260916.json"]["rows"] == 1
    assert by_source["runs/cfg-earley-fresh-typed-adjunct-repair-20260916.json"]["rows"] == 1
    assert by_source["runs/semantic-slot-adjunct-followup-20260916.json"]["rows"] == 1
    assert by_source["runs/inflection-clitic-distinct-suffix-repair-20260916.json"]["rows"] == 1
    assert by_source["runs/cfg-earley-fresh-opposing-adjunct-repair-20260916.json"]["rows"] == 1
    assert by_source["runs/inflection-clitic-distinct-possessive-repair-20260916.json"]["rows"] == 1
    assert by_source["runs/semantic-slot-final-locative-followup-20260916.json"]["rows"] == 1
    assert by_source["runs/cfg-earley-fresh-single-terminal-repair3-20260916.json"]["rows"] == 1
    assert by_source["runs/inflection-clitic-distinct-complementizer-repair-20260916.json"]["rows"] == 1
    assert by_source["runs/semantic-slot-temporal-followup-20260916.json"]["rows"] == 1
    assert by_source["runs/simultaneous-phrase-pair-constructor-20260916.json"]["rows"] == 1
    assert by_source["runs/centerout-observatory-scene-lattice-20260916.json"]["rows"] == 4
    assert by_source["runs/immutable-scene-exact-tape-resegment-20260916.json"]["rows"] == 1
    assert by_source["runs/paired-lexical-phrase-graph-live-emit-20260916.json"]["rows"] == 1
    assert by_source["runs/grammar-first-phrase-intersection-20260916.json"]["rows"] == 1
    assert by_source["runs/semantic-wordpair-event-graph-20260916.json"]["rows"] == 1
    assert by_source["runs/center-terminal-clause-family-20260916.json"]["rows"] == 8
    assert by_source["runs/grammar-first-matching-boundary-run-20260916.json"]["rows"] == 1
    assert by_source["runs/semantic-slot-attachment-repair-20260916-luna.json"]["rows"] == 2
    assert by_source["runs/finite-feature-center-grammar-20260916.json"]["rows"] == 3
    assert by_source["runs/joint-constituent-equation-scene-solver-20260916.json"]["rows"] == 9
    assert by_source["runs/word-internal-seam-equation-20260916.json"]["rows"] == 1
    assert by_source["runs/minimal-residual-grammar-20260916.json"]["rows"] == 4
    assert by_source["runs/semantic-relation-plan-solver-20260916.json"]["rows"] == 12
    assert by_source["runs/word-internal-seam-first-mismatch-repair-20260916.json"]["rows"] == 1
    assert by_source["runs/minimal-residual-grammar-repair-20260916.json"]["rows"] == 1
    assert by_source["runs/joint-constituent-equation-scene-repair-20260916.json"]["rows"] == 1
    assert by_source["runs/agreement-clitic-character-transducer-20260916.json"]["rows"] == 1
    assert by_source["runs/seed-benchmark-live-semantic-slot-expansion-20260916.json"]["rows"] == 1
    assert by_source["runs/semantic-valency-attachment-scene-lattice-20260916.json"]["rows"] == 6
    assert by_source["runs/agreement-clitic-outer-terminal-search-20260916.json"]["rows"] == 9
    assert by_source["runs/semantic-valency-clause-equation-solver-20260916.json"]["rows"] == 6
    assert by_source["runs/semantic-slot-exact-closure-frontier-20260916.json"]["rows"] == 1
    assert by_source["runs/outside-in-role-phrase-equation-20260916.json"]["rows"] == 1
    assert by_source["runs/cfg-earley-character-equation-forest-20260916.json"]["rows"] == 2
    assert by_source["runs/semantic-slot-first-residual-repair-20260916.json"]["rows"] == 6
    assert by_source["runs/char-lm-obligation-beam-20260916.json"]["rows"] == 64
    assert by_source["runs/center-free-clause-equation-ledger-20260916.json"]["rows"] == 1
    assert by_source["runs/inflectional-clitic-boundary-repair-20260916.json"]["rows"] == 10
    assert by_source["runs/dependency-seam-attachment-csp-20260916.json"]["rows"] == 64
    assert by_source["runs/center-residual-targeted-repair-20260917.json"]["rows"] == 8
    assert by_source["runs/dependency-role-first-residual-repair-20260917.json"]["rows"] == 5

    registry = json.loads((ROOT / "docs/experiment-novelty-registry.json").read_text())
    retained = {row["id"] for row in registry["entries"]}
    assert {
        "lexical-mirror-hypergraph-clause-20260916",
        "dialogue-relative-clause-csp-20260916",
        "function-word-boundary-dp-20260916",
        "lexical-mirror-hypergraph-edge-repair-20260916",
        "dialogue-relative-clause-followup-20260916",
        "function-word-boundary-single-repair-20260916",
        "lexical-mirror-hypergraph-verb-edge-repair-20260916",
        "dialogue-speaker-followup-20260916",
        "function-word-auxiliary-single-repair-20260916",
        "active-passive-attachment-csp-20260916",
        "semantic-slot-cross-boundary-dp-20260916",
        "bespoke-scene-lattice-free-center-20260916",
        "active-passive-attachment-followup-20260916",
        "bespoke-scene-lattice-single-slot-repair-20260916",
        "semantic-slot-single-object-repair-20260916",
        "finite-reverse-phrase-composition-bank-20260916",
        "centerfree-clause-pair-csp-20260916",
        "finite-semantic-palindrome-csp-20260916",
        "finite-reverse-phrase-single-boundary-repair-20260916",
        "centerfree-clause-pair-followup-20260916",
        "finite-semantic-csp-object-verb-repair-20260916",
        "cfg-earley-fresh-typed-adjunct-repair-20260916",
        "semantic-slot-adjunct-followup-20260916",
        "inflection-clitic-distinct-suffix-repair-20260916",
        "cfg-earley-fresh-opposing-adjunct-repair-20260916",
        "inflection-clitic-distinct-possessive-repair-20260916",
        "semantic-slot-final-locative-followup-20260916",
        "cfg-earley-fresh-single-terminal-repair3-20260916",
        "inflection-clitic-distinct-complementizer-repair-20260916",
        "semantic-slot-temporal-followup-20260916",
        "simultaneous-phrase-pair-constructor-20260916",
        "centerout-observatory-scene-lattice-20260916",
        "immutable-scene-exact-tape-resegment-20260916",
        "paired-lexical-phrase-graph-live-emit-20260916",
        "grammar-first-phrase-intersection-20260916",
        "semantic-wordpair-event-graph-20260916",
        "center-terminal-clause-family-20260916",
        "grammar-first-matching-boundary-run-20260916",
        "semantic-slot-attachment-repair-20260916-luna",
        "finite-feature-center-grammar-20260916",
        "joint-constituent-equation-scene-solver-20260916",
        "word-internal-seam-equation-20260916",
        "minimal-residual-grammar-20260916",
        "semantic-relation-plan-solver-20260916",
        "word-internal-seam-first-mismatch-repair-20260916",
        "minimal-residual-grammar-repair-20260916",
        "joint-constituent-equation-scene-repair-20260916",
        "agreement-clitic-character-transducer-20260916",
        "seed-benchmark-live-semantic-slot-expansion-20260916",
        "semantic-valency-attachment-scene-lattice-20260916",
        "agreement-clitic-outer-terminal-search-20260916",
        "semantic-valency-clause-equation-solver-20260916",
        "semantic-slot-exact-closure-frontier-20260916",
        "outside-in-role-phrase-equation-20260916",
        "cfg-earley-character-equation-forest-20260916",
        "semantic-slot-first-residual-repair-20260916",
        "char-lm-obligation-beam-20260916",
        "center-free-clause-equation-ledger-20260916",
        "inflectional-clitic-boundary-repair-20260916",
        "dependency-seam-attachment-csp-20260916",
        "center-residual-targeted-repair-20260917",
        "dependency-role-first-residual-repair-20260917",
    } <= retained


def test_registry_retains_new_lanes_and_keeps_shortcut_exclusions_separate():
    registry = json.loads((ROOT / "docs/experiment-novelty-registry.json").read_text())
    retained = {row["id"] for row in registry["entries"]}
    excluded = {row["id"] for row in registry["excluded"]}
    assert {
        "constrained-edit-program-constructor-20260916",
        "arbitrary-clause-macro-algebra-20260916",
        "discourse-relation-involution-20260916",
    } <= retained
    assert "inflection-clitic-boundary-search-20260916-luna-excluded" in excluded


def test_latest_three_lanes_keep_complete_prose_and_independent_audits():
    center = json.loads((ROOT / "runs/centerout-typed-semantic-debt-20260916.json").read_text())
    assert center["summary"] == {"candidate_count": 12, "exact_count": 0, "max_length": 109}
    assert all(row["provenance"]["catalogue_imported"] is False for row in center["rows"])
    assert all(row["audit"]["exact"] is False and row["audit"]["sha256_exact"] is False for row in center["rows"])
    assert all(row["next_repair"] for row in center["rows"])

    lexical = json.loads((ROOT / "runs/lexical-word-equation-grammar-intersection-20260916.json").read_text())
    assert len(lexical["candidates"]) == 4 and lexical["stats"]["exact"] == 0
    assert all(row["audit"]["rendered"].endswith(".") for row in lexical["candidates"])
    assert all(row["audit"]["two_pointer_exact"] is False for row in lexical["candidates"])
    assert all(row["audit"]["normalized_sha256"] != row["audit"]["reverse_sha256"] for row in lexical["candidates"])

    seed = json.loads((ROOT / "runs/seed-joint-slot-resegment-20260916.json").read_text())
    assert len(seed["candidates"]) == 9 and seed["stats"]["exact"] == 0
    assert all(" the old a faded" not in row["rendered"] for row in seed["candidates"])
    assert all(row["rendered"].endswith(".") and row["exact_audit"]["sha_equal"] is False for row in seed["candidates"])

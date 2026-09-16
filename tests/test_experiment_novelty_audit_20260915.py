import json
from pathlib import Path

from tools.audit_experiment_novelty_20260915 import audit


def test_novelty_audit_has_no_exact_collisions_and_catches_real_near_pairs():
    report = audit()
    assert report["exact_signature_collisions"] is False
    assert report["registered_entries"] >= 113
    assert report["excluded_routes"] == 13
    pairs = {(row["newer"], row["older"]) for row in report["near_pairs"]}
    # These are intentionally close families already in the historical ledger;
    # the report makes them visible so future work cannot silently replay them.
    assert ("connective-bearing-event-pair", "two-event-discourse-frame") in pairs
    assert ("dialogue-shared-topic-elliptical-residual", "dialogue-elliptical-ack-residual-inventory") in pairs


def test_latest_routes_are_not_flagged_as_near_duplicates():
    report = audit()
    by_id = {row["id"]: row for row in report["entries"]}
    assert by_id["semordnilap-template-inventory"]["manual_review_required"] is False
    assert by_id["corpus-sentence-gram-fst"]["manual_review_required"] is False
    assert by_id["grammar-boundary-resegmentation-repair"]["manual_review_required"] is False
    assert by_id["fixed-tape-valency-chart-repair"]["manual_review_required"] is False
    assert by_id["proper-name-caption-crossword"]["manual_review_required"] is False
    assert by_id["information-structure-focus-scope"]["manual_review_required"] is False
    assert by_id["anaphoric-scene-chain-composition"]["manual_review_required"] is False
    assert by_id["multiset-balanced-pair-sampling"]["manual_review_required"] is False
    assert by_id["proper-name-reverse-grammar"]["manual_review_required"] is False
    assert by_id["adaptive-crossword-span-cover"]["manual_review_required"] is False
    assert by_id["context-template-crossword-repair"]["manual_review_required"] is False
    assert by_id["syntactic-mirror-template-repair"]["manual_review_required"] is False
    assert by_id["semantic-selectional-prefix-automaton"]["manual_review_required"] is False
    assert by_id["model-authored-clause-bank-index"]["manual_review_required"] is False
    assert by_id["semantic-scene-seam-growth"]["manual_review_required"] is False
    assert by_id["live-seam-intent-continuation"]["manual_review_required"] is False
    assert by_id["role-aware-reversible-reservoir-centerout-20260915"]["manual_review_required"] is False
    assert by_id["variable-length-role-reservoir-centerout-20260915"]["manual_review_required"] is False
    assert by_id["asymmetric-template-reservoir-centerout-20260915"]["manual_review_required"] is False
    assert by_id["attested-phrase-pair-wrapper-20260915"]["manual_review_required"] is False
    assert by_id["homograph-sense-lattice-20260915"]["manual_review_required"] is False
    assert by_id["interrogative-quantifier-fsm-20260915"]["manual_review_required"] is False
    assert by_id["recursive-grammar-residual-dp-20260916"]["manual_review_required"] is False
    assert by_id["dialogue-speech-act-residual-20260916"]["manual_review_required"] is False
    assert by_id["brown-attested-residual-lattice-20260916"]["manual_review_required"] is False
    assert by_id["seedless-semantic-cfg-bilateral-20260916"]["manual_review_required"] is False
    assert by_id["whole-sentence-semordnilap-clauses-20260916"]["manual_review_required"] is False
    assert by_id["semantic-mutation-residual-20260916"]["manual_review_required"] is False
    assert by_id["rhetorical-plan-lattice-20260916"]["manual_review_required"] is False
    assert by_id["inflectional-fst-clitic-tape-20260916"]["manual_review_required"] is False
    assert by_id["induced-pcfg-character-equation-20260916"]["manual_review_required"] is False
    assert by_id["graph-to-prose-path-20260916"]["manual_review_required"] is False
    assert by_id["voice-alternation-residual-20260916"]["manual_review_required"] is False
    assert by_id["ccg-semantic-solver-20260916"]["manual_review_required"] is False
    assert by_id["dependency-completion-csp-20260916"]["manual_review_required"] is False
    assert by_id["lexical-word-equation-inventory-20260916"]["manual_review_required"] is False
    assert by_id["direct-constrained-authoring-20260916"]["manual_review_required"] is False
    assert by_id["lexical-chain-palindrome-20260916"]["manual_review_required"] is False
    assert by_id["morphology-semantic-template-csp-20260916"]["manual_review_required"] is False
    assert by_id["pivot-paragraph-beam-20260916"]["manual_review_required"] is False
    assert by_id["prosodic-foot-scene-constructor-20260916"]["manual_review_required"] is False
    assert by_id["induced-grammar-reverse-decoder-20260916"]["manual_review_required"] is False

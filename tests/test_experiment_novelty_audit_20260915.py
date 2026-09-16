import json
from pathlib import Path

from tools.audit_experiment_novelty_20260915 import audit


def test_novelty_audit_has_no_exact_collisions_and_catches_real_near_pairs():
    report = audit()
    assert report["exact_signature_collisions"] is False
    assert report["registered_entries"] == 61
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

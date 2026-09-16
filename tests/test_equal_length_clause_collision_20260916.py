import json
from pathlib import Path

from experiments.equal_length_clause_collision_20260916 import (
    EVIDENCE,
    EXPERIMENT_ID,
    SIGNATURE,
    build_clause_bank,
    clause_readability,
    run,
    two_pointer_verify,
)


def test_two_pointer_replay_is_independent_and_requires_equal_lengths():
    assert two_pointer_verify("abc", "cba")["closed"]
    result = two_pointer_verify("abcd", "cba")
    assert not result["closed"]
    assert not result["equal_length"]
    assert result["matched"] == 3


def test_banks_are_large_complete_semantic_clauses_in_normal_order():
    left = build_clause_bank("left", 256)
    right = build_clause_bank("right", 256)
    assert len(left) == len(right) == 256
    assert all(clause.words[0] in {"a", "the"} for clause in left + right)
    assert all(clause_readability(clause)["complete_svo_adjunct"] for clause in left + right)
    assert all(clause_readability(clause)["semantic_selection"] for clause in left + right)
    assert all(clause_readability(clause)["lexical_order_preserved"] for clause in left + right)
    left_content = {word for clause in left for word in clause.words if word not in {"a", "the", "in", "near", "under", "beside", "by", "around"}}
    right_content = {word for clause in right for word in clause.words if word not in {"a", "the", "at", "over", "behind", "within", "through", "among"}}
    assert not left_content & right_content


def test_packaged_collision_evidence_has_preflight_hash_and_admission_ledger():
    payload = run()
    assert payload["experiment_id"] == EXPERIMENT_ID
    assert payload["signature"] == SIGNATURE
    assert payload["novelty_preflight"]["performed_before_search"]
    assert not payload["novelty_preflight"]["exact_signature_collision"]
    assert not payload["novelty_preflight"]["exact_id_collision"]
    assert payload["config"]["equal_length_only"]
    assert payload["config"]["word_order_reversal"] is False
    assert payload["stats"]["left_bank"] >= 100_000
    assert payload["stats"]["right_bank"] >= 100_000
    assert payload["stats"]["exact"] == 0
    assert payload["stats"]["mechanically_admitted"] == 0
    assert payload["stats"]["max_probe_letters"] > 38
    assert payload["provenance"]["known_palindromes_used"] is False
    assert payload["repair"]["operator"]
    assert all(row["left_readability"]["complete_svo_adjunct"] for row in payload["rendered_probes"])
    assert all(row["right_readability"]["complete_svo_adjunct"] for row in payload["rendered_probes"])

    root = Path(__file__).resolve().parents[1]
    registry = json.loads((root / "docs/experiment-novelty-registry.json").read_text())
    row = next(item for item in registry["entries"] if item["id"] == EXPERIMENT_ID)
    assert row["signature"] == SIGNATURE
    assert (root / row["artifact"]).exists()
    assert all((root / item).exists() for item in row["run_artifacts"])
    assert EVIDENCE == root / row["run_artifacts"][0]

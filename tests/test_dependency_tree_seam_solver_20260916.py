import json
from pathlib import Path

from experiments.dependency_tree_seam_solver_20260916 import (
    ATTACHMENTS,
    EVIDENCE,
    EXPERIMENT_ID,
    SIGNATURE,
    build_tree_bank,
    run,
    solve_live_seam,
)


def test_complete_trees_record_agreement_valency_and_attachment_edges():
    left = build_tree_bank("left", 24)
    right = build_tree_bank("right", 24)
    assert len(left) == len(right) == 24
    assert {tree.attachment for tree in left} == set(ATTACHMENTS)
    assert all(tree.verb_form == "reads" for tree in left[:3])
    assert all(any(edge[0] == "nsubj" for edge in tree.dependencies) for tree in left + right)
    event = next(tree for tree in left if tree.attachment == "postverbal_event")
    obj = next(tree for tree in left if tree.attachment == "postverbal_object")
    assert ("obl", "verb", event.adjunct.location.word) in event.dependencies
    assert ("obl", "object", obj.adjunct.location.word) in obj.dependencies


def test_live_seam_dp_preserves_normal_order_and_reports_mismatch():
    left = build_tree_bank("left", 1)[0]
    right = build_tree_bank("right", 1)[0]
    seam = solve_live_seam(left, right)
    assert not seam["closed"]
    assert seam["dp_states"] >= 1
    assert seam["first_mismatch"] is not None
    assert left.words[0] == "the" and right.words[0] == "the"
    assert left.words != tuple(reversed(right.words))


def test_packaged_dependency_seam_evidence_is_registered_and_fail_closed():
    payload = run()
    assert payload["experiment_id"] == EXPERIMENT_ID
    assert payload["signature"] == SIGNATURE
    assert payload["novelty_preflight"]["performed_before_search"]
    assert not payload["novelty_preflight"]["exact_signature_collision"]
    assert not payload["novelty_preflight"]["exact_id_collision"]
    assert payload["config"]["hash_collision_only"] is False
    assert payload["config"]["reverse_decoding"] is False
    assert payload["stats"]["pair_states"] > 10_000
    assert payload["stats"]["exact"] == 0
    assert payload["stats"]["mechanically_admitted"] == 0
    assert payload["stats"]["max_probe_letters"] > 38
    assert payload["repair"]["operator"]
    assert all(row["left_tree"]["readability"]["complete_clause"] for row in payload["rendered_probes"])
    assert all(row["right_tree"]["readability"]["complete_clause"] for row in payload["rendered_probes"])
    root = Path(__file__).resolve().parents[1]
    registry = json.loads((root / "docs/experiment-novelty-registry.json").read_text())
    row = next(item for item in registry["entries"] if item["id"] == EXPERIMENT_ID)
    assert row["signature"] == SIGNATURE
    assert (root / row["artifact"]).exists()
    assert all((root / item).exists() for item in row["run_artifacts"])
    assert EVIDENCE == root / row["run_artifacts"][0]

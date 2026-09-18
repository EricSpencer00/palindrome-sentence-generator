"""Tests for Brown/wordfreq-backed residual discovery and proposal gating."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.brown_wordfreq_registry_partial_residual_scheduler_20260913 import (
    FrameSpec,
    VerbFrame,
    build_registry,
    discover_partial_contexts,
    group_discoveries,
    run,
)


def _registry():
    return build_registry()


def _proposal(discovery, registry):
    frame = next(frame for frame in registry["frames"] if frame.frame_id == discovery["frame_id"])
    return json.dumps({
        "proposal_id": "brown-registry-frozen-test",
        "context": discovery["context"],
        "required_reverse_prefix": discovery["required_reverse_prefix"],
        "locations": [{"place_det": "a", "place_adj": "quiet", "place": "studio"},
                      {"place_det": "a", "place_adj": "public", "place": "gallery"}],
        "purposes": [
            {"purpose": frame.purpose_frames[0].verb, "purpose_det": "the",
             "purpose_adj": discovery["discovered_from"]["purpose_adj"],
             "purpose_object": discovery["discovered_from"]["purpose_object"]},
            {"purpose": frame.purpose_frames[0].verb, "purpose_det": "the",
             "purpose_adj": "clear", "purpose_object": discovery["discovered_from"]["purpose_object"]},
        ],
    })


def test_registry_uses_brown_and_wordfreq_but_keeps_explicit_semantic_roles():
    registry = _registry()
    assert len(registry["entries"]) > 50
    assert registry["brown_role_counts"]["noun"] > 20
    assert registry["brown_role_counts"]["verb"] > 10
    assert registry["min_zipf"] == 2.2
    assert "artifact" in registry["entries"]["cart"].semantic_types
    assert registry["entries"]["cart"].brown_count > 0


def test_only_five_or_more_real_pairs_are_prompted():
    registry = _registry()
    discoveries = discover_partial_contexts(registry)
    assert discoveries
    assert min(row["matched_pairs"] for row in discoveries) >= 5
    assert all(row["required_reverse_prefix"] == row["opening_letters"][:row["matched_pairs"]]
               for row in discoveries)
    result = run(minimum_pairs=5)
    assert result["discovery_count"] == len(discoveries)
    assert result["prompt_group_count"] == len(group_discoveries(discoveries))
    assert all(row["matched_pairs"] >= 5 for row in result["prompted_contexts"])


def test_unindexed_authoring_token_does_not_create_a_prompt():
    registry = _registry()
    # This action/object/attachment is lexically present in the authored
    # frame, but its boundary signature does not match any five-letter seam.
    extra = FrameSpec("unindexed-read-book", "read", "book", "artifact", (
        # Every word is already registry-backed; Brown/POS and the endpoint
        # index decide whether this semantic relation becomes visible.
        VerbFrame("store", "artifact", ("paper",)),
        VerbFrame("cite", "artifact", ("paper",)),
    ))
    expanded = tuple(registry["frames"]) + (extra,)
    expanded_registry = dict(registry, frames=expanded)
    assert not any(row["frame_id"] == extra.frame_id for row in discover_partial_contexts(expanded_registry))


def test_discovered_context_accepts_only_registry_backed_frozen_proposals():
    registry = _registry()
    discovery = discover_partial_contexts(registry)[0]
    key = next(iter(group_discoveries([discovery])))
    result = run({key: (_proposal(discovery, registry),)})
    assert result["config"]["proposal_values_must_be_registry_backed"]
    assert len(result["frozen_responses"]) == 1
    evaluation = result["response_results"][0]
    assert evaluation["frozen_response"] and evaluation["preconstruction"]["discovered_context"]
    constructed = [row for row in evaluation["records"] if row.get("construction_started")]
    assert constructed
    assert all(row["independent_parse"]["ok"] for row in constructed)
    assert all(not row["outside_in_ledger"]["exact"] for row in constructed)
    assert not result["exact_candidates"]


def test_model_invented_word_is_rejected_before_assembly():
    registry = _registry(); discovery = discover_partial_contexts(registry)[0]
    key = next(iter(group_discoveries([discovery])))
    payload = json.loads(_proposal(discovery, registry))
    payload["purposes"][0]["purpose_object"] = "inventedword"
    result = run({key: (json.dumps(payload),)})
    row = result["response_results"][0]
    assert not row["preconstruction"]["accepted"]
    assert not row["construction_started"]
    assert row["rendered"] == []


def test_no_model_call_in_production_wrapper():
    result = run()
    assert result["config"]["model_calls_enabled"] is False
    assert result["provenance"]["external_model_calls"] == 0

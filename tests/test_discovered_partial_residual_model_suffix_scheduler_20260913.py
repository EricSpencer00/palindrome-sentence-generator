"""Tests for discovery-driven frozen proposal scheduling."""
import inspect
import json
from dataclasses import replace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.discovered_partial_residual_model_suffix_scheduler_20260913 import (
    AUTHORING_INVENTORY,
    FrameSpec,
    LocationSpec,
    PurposeSpec,
    discover_partial_contexts,
    group_discoveries,
    independent_parse,
    run,
)


def _proposal_for(discovery):
    frame = next(frame for frame in AUTHORING_INVENTORY if frame.frame_id == discovery["frame_id"])
    return json.dumps({
        "proposal_id": "frozen-discovery-test",
        "context": discovery["context"],
        "required_reverse_prefix": discovery["required_reverse_prefix"],
        "locations": [location.__dict__ for location in frame.locations],
        "purposes": [
            {"purpose": purpose.purpose, "purpose_det": "a", "purpose_adj": purpose.purpose_adjectives[0],
             "purpose_object": purpose.purpose_object}
            for purpose in frame.purposes[:2]
        ],
    })


def test_production_run_has_no_hand_selected_context_or_residual():
    source = inspect.getsource(run)
    assert "draw" not in source
    assert "ward" not in source
    assert "required_reverse_prefix =" not in source
    result = run()
    assert result["discovery_count"] > 0
    assert result["prompt_group_count"] > 0
    assert result["config"]["prompts_only_for_discovered_contexts"]


def test_every_prompt_group_is_derived_from_the_endpoint_index():
    discoveries = discover_partial_contexts()
    grouped = group_discoveries(discoveries)
    result = run()
    assert set(result["prompt_groups"]) == set(grouped)
    assert all(row["matched_pairs"] >= 2 for row in discoveries)
    assert all(row["required_reverse_prefix"] == row["opening_letters"][:row["matched_pairs"]]
               for row in discoveries)


def test_adding_unconnected_authoring_frame_creates_no_prompt_context():
    unconnected = FrameSpec(
        "unconnected-write-memo", "write", "memo", "artifact", ("new",),
        (LocationSpec("a", "quiet", "office"), LocationSpec("a", "public", "gallery")),
        (PurposeSpec("file", "paper", "artifact", ("final", "clear")),
         PurposeSpec("share", "paper", "artifact", ("final", "clear"))),
    )
    expanded = discover_partial_contexts(AUTHORING_INVENTORY + (unconnected,))
    assert not any(row["frame_id"] == unconnected.frame_id for row in expanded)
    assert all(row["frame_id"] != unconnected.frame_id
               for row in run(inventory=AUTHORING_INVENTORY + (unconnected,))["discovered_contexts"])


def test_new_prompt_context_exists_only_when_index_finds_a_real_residual():
    compatible = FrameSpec(
        "additional-paint-map", "paint", "canvas", "artifact", ("clean",),
        (LocationSpec("a", "quiet", "studio"), LocationSpec("a", "public", "gallery")),
        (PurposeSpec("display", "map", "artifact", ("clear",)),
         PurposeSpec("frame", "map", "artifact", ("simple",))),
    )
    expanded = discover_partial_contexts(AUTHORING_INVENTORY + (compatible,))
    assert any(row["frame_id"] == compatible.frame_id for row in expanded)
    assert any(key for key in group_discoveries(expanded)
               if json.loads(key)["frame_id"] == compatible.frame_id)


def test_only_a_frozen_response_for_a_discovered_key_reaches_evaluator():
    discovery = discover_partial_contexts()[0]
    key = next(key for key in group_discoveries(discover_partial_contexts())
               if json.loads(key)["frame_id"] == discovery["frame_id"]
               and json.loads(key)["required_reverse_prefix"] == discovery["required_reverse_prefix"]
               and json.loads(key)["object_adj"] == discovery["context"]["object_adj"])
    result = run({key: (_proposal_for(discovery),)})
    assert len(result["frozen_responses"]) == 1
    evaluation = result["response_results"][0]
    assert evaluation["frozen_response"] and evaluation["preconstruction"]["discovered_context"]
    assert evaluation["construction_started"]
    assert all(row["independent_parse"]["ok"] for row in evaluation["records"] if row.get("construction_started"))
    assert not result["exact_candidates"]


def test_unmatched_model_context_is_rejected_before_surface_assembly():
    discovery = discover_partial_contexts()[0]
    key = next(iter(group_discoveries(discover_partial_contexts())))
    payload = json.loads(_proposal_for(discovery))
    payload["context"]["frame_id"] = "not-discovered"
    result = run({key: (json.dumps(payload),)})
    row = result["response_results"][0]
    assert not row["preconstruction"]["accepted"]
    assert not row["construction_started"]
    assert row["rendered"] == []

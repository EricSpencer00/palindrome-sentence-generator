"""Tests for frozen partial-residual model proposals."""
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.partial_residual_model_suffix_proposal_20260913 import (
    ActionObjectContext,
    freeze_model_response,
    independent_parse,
    proposal_prompt,
    run,
)


def response(required="draw"):
    return json.dumps({
        "proposal_id": "typed-draw-attachments-01",
        "context": {"action": "draw", "object": "map"},
        "required_reverse_prefix": required,
        "locations": [
            {"place_det": "a", "place_adj": "quiet", "place": "studio"},
            {"place_det": "a", "place_adj": "public", "place": "gallery"},
        ],
        "purposes": [
            {"purpose": "protect", "purpose_det": "a", "purpose_adj": "kind", "purpose_object": "ward"},
            {"purpose": "label", "purpose_det": "a", "purpose_adj": "clear", "purpose_object": "map"},
        ],
    })


def test_prompt_requests_partial_typed_options_not_a_surface():
    prompt = proposal_prompt(ActionObjectContext("draw", "map"), "draw")
    assert "typed alternatives" in prompt
    assert "write a palindrome" not in prompt.lower()
    assert "complete sentence" in prompt


def test_unknown_surface_field_is_rejected_before_construction():
    payload = json.loads(response())
    payload["surface"] = "Draw a clear map in a quiet studio to protect a kind ward."
    with pytest.raises(ValueError):
        freeze_model_response(json.dumps(payload))


def test_malformed_external_output_is_still_frozen_and_never_constructed():
    result = run(("not json",))
    assert result["frozen_responses"][0]["raw_response"] == "not json"
    row = result["response_results"][0]
    assert row["frozen_response"] and row["model_output_is_proposal_only"]
    assert not row["construction_started"]


def test_valid_model_output_is_frozen_and_used_only_as_typed_proposals():
    result = run((response(),))
    assert result["config"]["model_calls_enabled"] is False
    assert result["frozen_responses"][0]["raw_response"] == response()
    assert result["frozen_responses"][0]["response_sha256"]
    result_row = result["response_results"][0]
    assert result_row["frozen_response"]
    assert result_row["model_output_is_proposal_only"]
    assert result_row["preconstruction"]["accepted"]
    assert result_row["construction_started"]
    # Python assembled this surface; the model response has no surface field.
    rendered = [row["rendered"] for row in result_row["records"] if row.get("rendered")]
    assert rendered
    assert all("Draw " in text for text in rendered)


def test_partial_residual_filters_attachments_before_python_assembly():
    result = run((response("draw"),))
    rows = result["response_results"][0]["records"]
    assert any(row.get("rejection") == "required_partial_residual_not_met" for row in rows)
    assert any(row.get("construction_started") for row in rows)
    assert all("letters" not in row.get("attachment", {}) for row in rows)


def test_wrong_required_residual_is_a_frozen_proposal_rejection():
    result = run((response("drat"),))
    row = result["response_results"][0]
    assert not row["preconstruction"]["accepted"]
    assert not row["construction_started"]
    assert not row["exact_candidates"]


def test_independent_reparse_and_ledger_gate_every_constructed_surface():
    result = run((response(),))
    rows = [row for row in result["response_results"][0]["records"] if row.get("construction_started")]
    assert rows
    assert all(independent_parse(row["rendered"])["ok"] for row in rows)
    assert all(row["outside_in_ledger"]["exact"] for row in result["exact_candidates"])
    assert all(row["mechanically_admitted"] for row in result["admitted_candidates"])
    # A partial match is diagnostic only: it cannot be promoted to exactness.
    assert all(not row["outside_in_ledger"]["exact"] for row in rows)

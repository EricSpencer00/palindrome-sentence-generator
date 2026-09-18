"""Tests for preconstruction screening and Python-owned ledger decisions."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.constrained_character_ledger_authoring_20260913 import (
    ModelResponse, authored_proposals, construct_candidate, outside_in_ledger,
    preconstruction_gate, run,
)


def test_known_response_is_rejected_before_surface_construction():
    base = authored_proposals()[0]
    response = ModelResponse(base.proposal_id, base.frame, base.role_options,
                             base.verb_object_types, base.verb_subject_type,
                             verbatim_text="A man a plan a canal Panama")
    gate = preconstruction_gate(response)
    assert not gate["accepted"]
    assert not gate["construction_started"]
    row = construct_candidate(response, tuple(slot[0] for slot in response.choices))
    assert row["rendered"] is None and row["construction_started"] is False


def test_ledger_mismatch_cannot_be_emitted_as_candidate():
    response = authored_proposals()[0]
    words = ("a", "careful", "artist", "repairs", "a", "brief", "canvas", "in", "a", "quiet", "studio")
    row = construct_candidate(response, words)
    assert row["construction_started"]
    assert not row["outside_in_ledger"]["exact"]
    assert not row["mechanically_admitted"]
    assert "outside_in_ledger_mismatch" in row["rejection_codes"]


def test_ledger_reports_first_actual_mismatch_without_reflection():
    ledger = outside_in_ledger("A calm artist repairs a clean model.")
    assert not ledger["exact"]
    assert ledger["first_mismatch"]["left_index"] == 0
    assert ledger["first_mismatch"]["left"] != ledger["first_mismatch"]["right"]


def test_run_keeps_exact_and_admitted_surfaces_separate():
    result = run(max_combinations=1_000)
    assert result["config"]["python_owns_outside_in_exact_ledger"]
    assert result["config"]["preconstruction_catalogue_gate"]
    assert result["all_proposals_screened"] and result["proposal_count"] == len(result["proposal_screening"])
    assert all(row["outside_in_ledger"]["exact"] for row in result["exact_candidates"])
    assert all(row["mechanically_admitted"] for row in result["admitted_candidates"])

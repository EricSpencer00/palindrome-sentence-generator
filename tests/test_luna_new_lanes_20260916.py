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
    assert report["candidate_count"] == 4657
    assert report["exact_count"] == 79
    by_source = {row["source_run"]: row for row in report["route_summary"]}
    assert by_source["runs/constrained-edit-program-constructor-20260916.json"]["rows"] == 4
    assert by_source["runs/arbitrary-clause-macro-algebra-20260916.json"]["rows"] == 4
    assert by_source["runs/discourse-relation-involution-20260916.json"]["rows"] == 12


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

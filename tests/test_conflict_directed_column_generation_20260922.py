import hashlib
import json
from pathlib import Path

from experiments.conflict_directed_column_generation_20260922 import (
    EXPERIMENT_ID,
    PREFLIGHT_SIGNATURE,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments" / "conflict_directed_column_generation_20260922.py"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID / "search.json"


def payload():
    return json.loads(ARTIFACT.read_text())


def test_remote_cp_sat_run_is_source_identical_and_bounded():
    row = payload()
    assert row["preflight_signature"] == PREFLIGHT_SIGNATURE
    assert row["provenance"]["host"] == "hst-bench"
    assert row["solver"]["engine"] == "OR-Tools CP-SAT"
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == row["provenance"]["committed_source_sha256"]
    assert row["provenance"]["source_sha256"] == (
        "bc5dcfc4f4a78330b519e3fef2e62fb908204d2eeb5ab00992e3d00824d65899"
    )
    assert row["bounds"] == {
        "scene_plans": 2,
        "max_iterations_per_plan": 8,
        "max_reparsed_columns_per_core": 2,
        "vocabulary": "fixed common-word pools declared in source",
    }
    assert len(row["plans"]) == 2


def test_every_starting_path_is_complete_and_semantically_typed():
    for plan in payload()["plans"]:
        assert plan["state_schema"] == [
            "scene_plan", "complete_clause_skeletons", "packed_constituent_DAG",
            "feature_env", "selected_column_per_slot", "char_offsets",
            "mirror_equalities", "nogoods",
        ]
        control = plan["initial_complete_control"]
        assert control["parse"]["accepted"]
        assert control["audit"]["letters"] > 38
        assert len(control["parse"]["clauses"]) == 2
        assert all(control["parse"]["checks"].values())


def test_conflicts_are_minimum_and_drive_only_contextual_columns():
    for plan in payload()["plans"]:
        assert 1 <= len(plan["iterations"]) <= 8
        assert plan["final_obstruction"] or plan["survivor"]
        for iteration in plan["iterations"]:
            if "unsat_core" not in iteration:
                continue
            core = iteration["unsat_core"]
            assert core["minimum_cardinality"] == len(core["minimum_positions"])
            assert all(check["feasible"] for check in core["proper_subset_checks"])
            response = iteration["oracle_response"]
            assert len(response["added_column_ids"]) <= 2
            for item in response["reparse_evidence"]:
                assert item["slot"] in core["implicated_slots"]
                assert item["full_plan_accepted"]
                assert item["full_clause_surface"].endswith(".")
                assert all(item["checks"].values())
        assert all(item["reparsed_complete_clause"] for item in plan["added_columns"])
        requested = {slot for iteration in plan["iterations"] if "unsat_core" in iteration
                     for slot in iteration["unsat_core"]["implicated_slots"]}
        assert all(item["slot"] in requested for item in plan["added_columns"])


def test_nogoods_and_survivor_or_obstruction_are_persisted_without_shortcuts():
    row = payload()
    for plan in row["plans"]:
        responses_with_additions = sum(
            bool(iteration.get("oracle_response", {}).get("added_column_ids"))
            for iteration in plan["iterations"]
        )
        assert len(plan["nogoods"]) == responses_with_additions
        for nogood in plan["nogoods"]:
            assert nogood["slots"]
            assert nogood["old_domain_sizes"]
        if plan["survivor"]:
            survivor = plan["survivor"]
            assert survivor["independent_exact_audit"]["letters"] > 38
            assert survivor["independent_exact_audit"]["two_pointer_exact"]
            assert survivor["independent_exact_audit"]["sha_equal"]
            assert survivor["parse"]["accepted"]
            assert all(survivor["central_admission"].values())
        else:
            assert plan["final_obstruction"]["core"]
    provenance = row["provenance"]
    for key in ("seed_text_used", "catalogue_text_used", "local_r_equals_s_family_used",
                "finished_mirror_units_used", "fragments_used", "post_hoc_repair_used",
                "complete_sentence_bank_materialized"):
        assert provenance[key] is False

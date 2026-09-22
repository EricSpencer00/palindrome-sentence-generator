import hashlib
import json
from pathlib import Path

from experiments.conflict_directed_column_generation_20260922 import (
    CharacterTrie,
    Column,
    EXPERIMENT_ID,
    OracleEntry,
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
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == row["provenance"]["source_sha256"]
    assert row["bounds"] == {
        "scene_plans": 2,
        "max_iterations_per_plan": 8,
        "max_reparsed_columns_per_core": 2,
        "vocabulary": "fixed pre-solve common Brown/WordNet character tries",
        "max_oracle_entries_per_slot": 512,
        "max_query_terminals": 32,
    }
    assert len(row["plans"]) == 2
    oracle = row["lexical_oracle"]
    assert oracle["construction"] == "fixed pre-solve character tries by existing plan and slot"
    assert oracle["proper_name_tags_allowed"] is False
    assert len(oracle["inventory_sha256"]) == 64
    assert len(oracle["brown_index_sha256"]) == 64


def test_character_trie_enforces_length_offset_and_required_character():
    entries = tuple(
        OracleEntry(Column(f"p:s:{word}", "s", word, "np", lemma=word),
                    word, count, ("NN",), (f"{word}.n.01",))
        for word, count in (("stone", 8), ("stare", 5), ("store", 7), ("at", 20))
    )
    result = CharacterTrie(entries).query(length=5, local_offset=2,
                                          required_chars=("o",))
    assert [entry.column.text for entry in result["entries"]] == ["stone", "store"]
    assert result["terminals_matching"] == 2


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
            assert response["core_targeted_only"]
            assert len(response["queries"]) == 2 * len(core["minimum_positions"])
            core_offsets = {
                (side["slot"], side["local_offset"])
                for position in core["minimum_positions"]
                for side in (position["left"], position["right"])
            }
            accounted = response["reparse_evidence"] + response["rejections"]
            for query in response["queries"]:
                assert (query["slot"], query["local_offset"]) in core_offsets
                assert query["returned_column_ids"] == [
                    item["column_id"] for item in accounted
                    if item["query_id"] == query["query_id"]
                ]
                for item in accounted:
                    if item["query_id"] != query["query_id"]:
                        continue
                    assert len(item["tape"]) == query["required_tape_length"]
                    assert item["tape"][query["local_offset"]] in (
                        query["required_chars_from_active_mirror_domain"]
                    )
            for item in response["reparse_evidence"]:
                assert item["slot"] in core["implicated_slots"]
                assert item["full_clause_surface"].endswith(".")
                assert all(item["full_clause_checks"].values())
                if item["decision"] == "add":
                    assert item["full_plan_accepted"]
                    assert all(item["checks"].values())
            assert set(response["added_column_ids"]) == {
                item["column_id"] for item in response["reparse_evidence"]
                if item["decision"] == "add"
            }
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

from __future__ import annotations

import json
import re
from pathlib import Path

from experiments.incumbent_672_discourse_linked_reverse_chain_20260922 import (
    OUT,
    SNAPSHOT,
    build_payload,
    independent_audit,
    normalize,
)


ROOT = Path(__file__).resolve().parents[1]


def test_generated_chain_is_exact_and_uses_its_actual_length() -> None:
    payload = build_payload()
    row = payload["rows"][0]
    saved = json.loads(OUT.read_text())
    assert row["rendered"] == saved["rows"][0]["rendered"]
    assert payload["stats"] == saved["stats"]
    left = " ".join(item["surface"] for item in row["left_chain"])
    right = " ".join(item["surface"] for item in row["right_chain_rendered_order"])
    assert payload["parent"]["letters"] == 568
    actual_length = row["independent_audit"]["normalized_letters"]
    assert actual_length > 648
    assert row["growth_over_parent"] == actual_length - 568
    assert len(normalize(left)) == len(normalize(right))
    assert normalize(left) == normalize(right)[::-1]
    assert row["id"].endswith(f"-{actual_length}")
    assert row["independent_audit"]["two_pointer_exact"] is True
    assert row["independent_audit"]["sha_equal"] is True
    assert row["audit"]["project_validator_exact"] is True
    assert row["source_composition"]["parent_outside_seam_byte_for_byte"] is True
    assert row["source_composition"]["left_inserted_surface"] == left
    assert row["source_composition"]["right_inserted_surface"] == right


def test_online_scene_joins_and_rejection_gates_are_recorded() -> None:
    row = json.loads(OUT.read_text())["rows"][0]
    left = row["left_chain"]
    right = row["right_chain_rendered_order"]
    assert len(left) == len(right) == 4
    for chain in (left, right):
        for before, after in zip(chain, chain[1:]):
            assert before["surface"].split()[2].rstrip(".") == after["surface"].split()[0]
        assert all(item["surface"].endswith(".") for item in chain)
        assert all(item["surface"].split()[0] != item["surface"].split()[2].rstrip(".") for item in chain)
    search = row["online_search"]
    assert search["right_chain_discovered_tail_first"] is True
    assert search["left_subject_follows_previous_object"] is True
    assert search["right_subject_follows_previous_object"] is True
    assert search["states_examined"] > 0
    assert search["attempts"]
    assert search["residual_scope"] == "one complete clause at a time"
    assert search["cross_clause_character_residual"] is False
    assert row["novelty"]["snapshot_chronology"].startswith("retrospective")
    assert row["novelty"]["all_inserted_relations_absent_in_parent_commit"] is True
    assert row["novelty"]["pre_search_archive_scan"]["match_count"] == 0
    assert row["novelty"]["pre_search_archive_scan"]["tracked_json_files"] > 0
    assert row["novelty"]["failed_attempts_scanned"] is True
    assert row["provenance"]["preauthored_pair_catalogue"] is False
    assert row["provenance"]["human_certified"] is False
    assert row["global_gate"]["coherent_scene_certified"] is False


def test_frozen_global_snapshot_is_the_novelty_source() -> None:
    payload = json.loads(SNAPSHOT.read_text())
    assert payload["snapshot_id"] == "incumbent-672-global-novelty-snapshot-20260922"
    assert payload["file_count"] == payload["manifest_entry_count"]
    assert payload["total_bytes"] > 0
    assert payload["descendants_scanned_for_global_novelty"] is True
    assert payload["descendants_counted_as_568_parent_evidence"] is False
    assert "failed attempt_rendered" in payload["scope"]
    row = json.loads(OUT.read_text())["rows"][0]
    assert row["novelty"]["manifest_sha256"] == payload["manifest_sha256"]
    assert set(row["novelty"]["rejected_prior_edges"]) == set()


def test_local_vector_is_only_an_independent_control() -> None:
    left = "Nora sees Leon. Leon stops Aron. Aron sees Noel. Noel sees Mara."
    right = "Aram sees Leon. Leon sees Nora. Nora spots Noel. Noel sees Aron."
    assert normalize(left) == normalize(right)[::-1]
    assert independent_audit(left + " " + right)["normalized_letters"] == 98
    generated = json.loads(OUT.read_text())["rows"][0]
    generated_left = " ".join(item["surface"] for item in generated["left_chain"])
    assert generated_left != left

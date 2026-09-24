from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.dpo_readability_judge import prepare


ROOT = Path(__file__).resolve().parents[1]


def packet_with_tasks(n: int) -> dict:
    return {
        "rater_form": {
            "items": [
                {
                    "task_id": f"pair-{i:03d}",
                    "a": {"text": f"Passage A number {i} reads clearly."},
                    "b": {"text": f"Number {i} passage reads clearly A."},
                }
                for i in range(n)
            ]
        },
        # Deliberately include labels the data builder must never inspect.
        "answer_key": [{"task_id": "pair-000", "condition": "candidate"}],
    }


def responses_for(packet: dict) -> dict:
    task_ids = [row["task_id"] for row in packet["rater_form"]["items"]]
    return {
        "raters": [
            {"rater_id": f"R{i:03d}", "answers": [
                {"task_id": task_id, "prefer": "A" if i % 2 else "B"}
                for task_id in task_ids
            ]}
            for i in range(3)
        ]
    }


def test_prepares_blind_dpo_pairs_and_splits_by_task() -> None:
    packet = packet_with_tasks(100)
    data = prepare(packet, responses_for(packet), seed=4)

    assert data["format"] == "palindrome-readability-dpo-v1"
    assert data["status"] == "ready_for_dpo_training"
    assert data["counts"]["independent_raters"] == 3
    assert data["counts"]["train_task_ids"] >= 20
    assert data["counts"]["validation_task_ids"] >= 8
    assert data["training_data"] and data["validation_data"]
    assert all(set(row) == {"prompt", "chosen", "rejected"}
               for row in data["training_data"] + data["validation_data"])
    assert all(row["chosen"] in ("A", "B") and row["rejected"] != row["chosen"]
               for row in data["training_data"] + data["validation_data"])

    folds = data["split_manifest"]["task_folds"]
    for task_id in folds:
        assert folds[task_id] in ("train", "validation")
    assert set(data["split_manifest"]["training_task_ids"]).isdisjoint(
        data["split_manifest"]["validation_task_ids"])


def test_empty_existing_reader_packet_is_not_misreported_as_a_verifier() -> None:
    packet_path = ROOT / "runs" / "reader-package-v4-20260919.json"
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    data = prepare(packet, {"raters": []})

    assert data["status"] == "insufficient_reader_preferences"
    assert data["counts"]["preference_examples"] == 0
    assert data["training_data"] == []
    assert data["validation_data"] == []
    assert any("not an exactness or readability certificate" in limit
               for limit in data["limits"])


def test_ties_are_skipped_and_duplicates_are_rejected() -> None:
    packet = packet_with_tasks(2)
    responses = {"raters": [{"rater_id": "R001", "answers": [
        {"task_id": "pair-000", "prefer": "tie"},
        {"task_id": "pair-001", "prefer": "A"},
    ]}]}
    data = prepare(packet, responses)
    assert data["counts"]["preference_examples"] == 1
    assert data["counts"]["skipped"]["tie_or_unsure"] == 1

    duplicate = {"raters": [{"rater_id": "R001", "answers": [
        {"task_id": "pair-000", "prefer": "A"},
        {"task_id": "pair-000", "prefer": "B"},
    ]}]}
    with pytest.raises(ValueError, match="duplicate response"):
        prepare(packet, duplicate)


def test_unknown_task_and_malformed_choices_fail_closed() -> None:
    packet = packet_with_tasks(1)
    with pytest.raises(ValueError, match="unknown task"):
        prepare(packet, {"raters": [{"rater_id": "R001", "answers": [
            {"task_id": "not-in-packet", "prefer": "A"},
        ]}]})
    with pytest.raises(ValueError, match="invalid preference"):
        prepare(packet, {"raters": [{"rater_id": "R001", "answers": [
            {"task_id": "pair-000", "prefer": "candidate"},
        ]}]})


def test_reused_passages_cannot_leak_across_preference_folds() -> None:
    packet = {
        "rater_form": {"items": [
            {"task_id": "pair-a", "a": {"text": "Shared passage."},
             "b": {"text": "Different passage one."}},
            {"task_id": "pair-b", "a": {"text": "Shared passage."},
             "b": {"text": "Different passage two."}},
        ]}
    }
    responses = {"raters": [{"rater_id": "R001", "answers": [
        {"task_id": "pair-a", "prefer": "A"},
        {"task_id": "pair-b", "prefer": "B"},
    ]}]}
    result = prepare(packet, responses, seed=1)
    folds = result["split_manifest"]["task_folds"]
    assert folds["pair-a"] == folds["pair-b"]
    assert result["counts"]["train_independent_groups"] + result["counts"][
        "validation_independent_groups"] == 1

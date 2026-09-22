from collections import Counter
import json
from pathlib import Path

from experiments.freshness_register_reader_packet_20261002 import (
    build_payloads,
    words,
)


ROOT = Path(__file__).resolve().parents[1]


def test_target_is_exact_and_mechanically_admitted() -> None:
    _, key = build_payloads()
    target = key["target"]

    assert target["audit"]["letters"] == 42
    assert target["audit"]["two_pointer_exact"]
    assert target["audit"]["sha256_forward"] == (
        "1013004f658bdefeaaf7dea69c6d90a5d2c53381dbb5d7290ab7b25a8e5de1c3"
    )
    assert all(target["mechanical_checks"].values())


def test_packet_is_blinded_randomized_and_has_intact_controls() -> None:
    rater, key = build_payloads()

    assert len(rater["items"]) == 8
    assert [row["order"] for row in rater["items"]] == list(range(1, 9))
    assert all(set(row) == {"item_id", "order", "text"} for row in rater["items"])
    assert key["design"]["condition_hidden_from_rater"]
    assert key["design"]["order_randomized"]
    assert key["design"]["intact_prose_control_count"] == 3
    assert key["design"]["matched_control_shuffle_count"] == 3
    assert not key["design"]["programmatic_scores_certify_readability"]


def test_every_shuffle_preserves_its_matched_word_multiset() -> None:
    rater, key = build_payloads()
    text_by_id = {row["item_id"]: row["text"] for row in rater["items"]}
    key_by_source = {row["source_id"]: row for row in key["items"]}

    for row in key["items"]:
        if row["condition"] != "shuffled":
            continue
        mate = key_by_source[row["matched_pair_source_id"]]
        assert Counter(words(text_by_id[row["item_id"]])) == Counter(
            words(text_by_id[mate["item_id"]])
        )


def test_checked_in_packet_matches_generator() -> None:
    rater, key = build_payloads()

    assert json.loads(
        (ROOT / "runs" / "freshness-register-reader-packet-20261002.rater.json").read_text()
    ) == rater
    assert json.loads(
        (ROOT / "runs" / "freshness-register-reader-packet-20261002.key.json").read_text()
    ) == key

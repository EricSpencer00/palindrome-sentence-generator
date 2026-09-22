from collections import Counter
import json
from pathlib import Path

from experiments.freshness_register_reader_packet_20261002 import words
from experiments.freshness_register_reader_packet_v2_20261002 import build_payloads


ROOT = Path(__file__).resolve().parents[1]


def test_both_targets_are_exact_and_gate_clean_but_not_reader_certified() -> None:
    _, key = build_payloads()
    targets = {row["audit"]["letters"]: row for row in key["targets"]}

    assert set(targets) == {42, 44}
    assert targets[42]["audit"]["sha256_forward"] == (
        "1013004f658bdefeaaf7dea69c6d90a5d2c53381dbb5d7290ab7b25a8e5de1c3"
    )
    assert targets[44]["audit"]["sha256_forward"] == (
        "96462b7bc06958668e9d13d13ebd0b63e4f682a51939c5bc34d7aa230d39b104"
    )
    assert all(row["audit"]["two_pointer_exact"] for row in targets.values())
    assert all(all(row["mechanical_checks"].values()) for row in targets.values())
    assert not any(row["reader_certified"] for row in targets.values())


def test_v2_blinds_conditions_and_pairs_every_shuffle() -> None:
    rater, key = build_payloads()
    text_by_id = {row["item_id"]: row["text"] for row in rater["items"]}
    key_by_source = {row["source_id"]: row for row in key["items"]}

    assert len(rater["items"]) == 10
    assert all(set(row) == {"item_id", "order", "text"} for row in rater["items"])
    assert key["design"]["fragment_completeness_question_included"]
    assert rater["status"] == "closed_pre_reader_parse_gate"
    assert rater["do_not_administer"]
    assert not key["design"]["pre_reader_parse_gate_passed"]
    assert not key["design"]["administered"]
    for row in key["items"]:
        if row["condition"] != "shuffled":
            continue
        mate = key_by_source[row["matched_pair_source_id"]]
        assert Counter(words(text_by_id[row["item_id"]])) == Counter(
            words(text_by_id[mate["item_id"]])
        )


def test_checked_in_v2_packet_matches_generator() -> None:
    rater, key = build_payloads()
    assert json.loads(
        (ROOT / "runs" / "freshness-register-reader-packet-v2-20261002.rater.json").read_text()
    ) == rater
    assert json.loads(
        (ROOT / "runs" / "freshness-register-reader-packet-v2-20261002.key.json").read_text()
    ) == key

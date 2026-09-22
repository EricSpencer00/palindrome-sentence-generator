import hashlib
import json
import re

from experiments.incumbent_498_partial_boundary_frontier_20261002 import (
    PARENT_SHA256,
    build_payload,
)
from llm_palindrome.validator import is_palindrome


def letters(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())


def test_loads_the_exact_498_content_parent_and_excludes_530_control():
    payload = build_payload()
    parent = payload["parent"]
    value = letters(parent["rendered"])
    assert parent["letters"] == len(value) == 498
    assert hashlib.sha256(value.encode()).hexdigest() == PARENT_SHA256
    assert value == value[::-1]
    assert payload["excluded_control"]["letters"] == 530
    assert payload["excluded_control"]["used_as_parent"] is False


def test_every_saved_child_is_independently_exact_and_over_530():
    payload = build_payload()
    assert payload["stats"]["independently_exact_children"] == 6
    assert payload["stats"]["children_over_530"] == 6
    for row in payload["rows"]:
        value = letters(row["rendered"])
        assert len(value) == row["audit"]["letters"] > 530
        assert value == value[::-1]
        assert is_palindrome(row["rendered"])
        assert row["audit"]["sha256_forward"] == hashlib.sha256(value.encode()).hexdigest()
        assert row["audit"]["sha256_forward"] == row["audit"]["sha256_reverse"]
        assert row["live_state"]["final_owner"] is None
        assert row["live_state"]["final_residual"] == ""


def test_repair_changes_the_incumbent_cursor_and_closes_both_partial_words():
    payload = build_payload()
    repaired = [row for row in payload["rows"] if row["seam_id"] == "depth54-repair"]
    assert len(repaired) == 3
    assert payload["incumbent_specific_repair"]["cursor_delta"] == 8
    for row in repaired:
        assert row["live_state"]["left_cursor"] == 54
        assert row["live_state"]["right_cursor"] == 444
        assert row["live_state"]["left_partial_word"] == "m|any -> poem | any (new word boundary)"
        assert row["live_state"]["right_partial_word"] == "na|me -> name"
        assert "A poem. Any." in row["rendered"]
        assert "my name. Opa," in row["rendered"]


def test_frontier_adds_lexical_and_event_content():
    payload = build_payload()
    assert len(payload["active_frontier"]) == 3
    for row in payload["rows"]:
        assert "ram" in row["new_lexical_content"]
        assert row["new_event_content"]
        assert row["reader_status"].startswith("not promoted")


def test_serialized_artifact_matches_runtime_payload():
    payload = build_payload()
    artifact = json.loads(
        open("runs/incumbent-498-partial-boundary-frontier-20261002.json").read()
    )
    assert artifact == payload

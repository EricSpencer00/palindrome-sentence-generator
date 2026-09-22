import hashlib
import json
import re

from experiments.incumbent_498_note_seam_growth_20261002 import build_payload
from llm_palindrome.validator import is_palindrome


def letters(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())


def test_note_set_seam_is_live_and_exact():
    payload = build_payload()
    seam = payload["live_seam"]
    assert seam["left_cursor"] == 88
    assert seam["right_cursor"] == 410
    assert seam["retained_letters"] == 322
    assert seam["boundary_tapes"] == ["anote", "etona"]
    assert seam["reverse_exact"] is True
    assert seam["partial_word"] == "s|et"


def test_all_children_are_independently_exact_and_above_control():
    payload = build_payload()
    assert payload["stats"]["independently_exact_children"] == 3
    assert payload["stats"]["children_over_530"] == 3
    assert payload["excluded_control"]["used_as_parent"] is False
    for row in payload["rows"]:
        value = letters(row["rendered"])
        assert len(value) == row["audit"]["letters"] > 530
        assert value == value[::-1]
        assert is_palindrome(row["rendered"])
        assert hashlib.sha256(value.encode()).hexdigest() == row["audit"]["sha256_forward"]
        assert row["audit"]["sha256_forward"] == row["audit"]["sha256_reverse"]
        assert row["live_state"]["final_owner"] is None
        assert row["live_state"]["final_residual"] == ""


def test_new_seam_removes_prior_worst_opening_and_adds_content():
    payload = build_payload()
    assert payload["stats"]["additional_parent_letters_removed_per_side"] == 34
    for row in payload["rows"]:
        assert "A poem. Any. Me?" not in row["rendered"]
        assert "A note: State: 'Go two.' 'No, do two.'" in row["rendered"]
        assert "at Set on a" in row["rendered"]
        assert {"note", "ram", "set"} <= set(row["new_lexical_content"])


def test_serialized_artifact_matches_runtime_payload():
    payload = build_payload()
    artifact = json.loads(open("runs/incumbent-498-note-seam-growth-20261002.json").read())
    assert artifact == payload

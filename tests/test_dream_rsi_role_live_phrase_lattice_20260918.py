import json
from pathlib import Path

from experiments.dream_rsi_role_live_phrase_lattice_20260918 import (
    parse_items,
    role_path_possible,
)


def test_parse_items_keeps_only_valid_role_tagged_units():
    raw = (
        '{"items":['
        '{"text":"the archivist","role":"subject"},'
        '{"text":"opens the ledger","role":"verb_phrase"},'
        '{"text":"the ledger","role":"object"},'
        '{"text":"at dawn","role":"adjunct"},'
        '{"text":"not valid!","role":"object"},'
        '{"text":"the archivist","role":"subject"}'
        ']}'
    )
    items, error = parse_items(raw)
    assert error is None
    assert [item["text"] for item in items] == [
        "the archivist", "opens the ledger", "the ledger", "at dawn"
    ]


def test_role_path_rejects_object_before_subject_or_verb():
    role_map = {
        "the ledger": "object",
        "opens the ledger": "verb_phrase",
        "the archivist": "subject",
    }
    assert role_path_possible(("the archivist", "opens the ledger", "the ledger"), role_map)
    assert not role_path_possible(("the ledger", "opens the ledger"), role_map)


def test_saved_run_records_live_grammar_failure_without_claiming_readability():
    path = Path("runs/dream-rsi-role-live-phrase-lattice-20260918.json")
    payload = json.loads(path.read_text())
    assert payload["provenance"]["human_readability_certified"] is False
    assert payload["reader_gate"]["status"] == "closed"
    assert payload["stats"]["exact"] == 0
    assert payload["stats"]["syntax_rejections"] > 0

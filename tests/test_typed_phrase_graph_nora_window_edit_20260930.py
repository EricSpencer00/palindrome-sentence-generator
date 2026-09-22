import json
from pathlib import Path

from llm_palindrome.validator import is_palindrome, normalize


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/typed-phrase-graph-nora-window-edit-20260930.json"


def test_selected_nora_aron_pair_is_exact_and_provenanced():
    data = json.loads(RUN.read_text())
    row = data["candidate"]
    assert row["left"] == "Nora, did I live?"
    assert row["right"] == "Evil, I did, Aron."
    assert row["window_tape_reverse"]
    assert row["outside_tape_preserved"]
    assert is_palindrome(row["text"])
    assert normalize(row["text"]) == normalize(row["text"])[::-1]
    assert row["audit"]["letters"] == 232


def test_nearby_variants_are_recorded_without_silent_admission():
    data = json.loads(RUN.read_text())
    assert len(data["variants"]) == 5
    assert data["provenance"]["posthoc_character_repair"] is False

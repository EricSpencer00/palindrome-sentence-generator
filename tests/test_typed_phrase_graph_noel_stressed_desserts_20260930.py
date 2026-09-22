import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "runs/typed-phrase-graph-noel-stressed-desserts-edit-20260930.json"


def test_paired_edit_is_exact_and_240_letters():
    d = json.loads(ART.read_text())
    text = d["candidate"]["text"]
    letters = re.sub(r"[^a-z]", "", text.lower())
    assert len(letters) == 240
    assert letters == letters[::-1]
    assert d["candidate"]["audit"]["sha_equal"]
    assert d["outside_tape_preserved"]


def test_only_declared_windows_changed():
    d = json.loads(ART.read_text())
    assert d["window_diff"]["old"] == ["Noel, I saw stressed.", "Desserts was I, Leon."]
    assert d["window_diff"]["new"] == ["Noel, was I stressed?", "Desserts, I saw, Leon."]

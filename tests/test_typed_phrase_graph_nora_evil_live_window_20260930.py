import json
from pathlib import Path

from llm_palindrome.validator import is_palindrome, normalize


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "runs/typed-phrase-graph-nora-evil-live-window-20260930.json"


def test_window_lane_has_independent_exact_candidate_and_preserves_outer_tape():
    data = json.loads(ARTIFACT.read_text())
    selected = data["selected"]
    assert selected["window"] == {
        "left": "Nora, was I evil?",
        "right": "Live, I saw, Aron.",
    }
    tape = normalize(selected["candidate"])
    assert selected["audit"]["two_pointer_exact"]
    assert selected["audit"]["validator_exact"]
    assert selected["audit"]["sha_equal"]
    assert tape == tape[::-1]
    assert selected["outside_tape_preserved"]


def test_every_rendered_row_records_joint_obligation_and_audit():
    data = json.loads(ARTIFACT.read_text())
    assert len(data["rendered_candidates"]) == 4
    for row in data["rendered_candidates"]:
        assert "joint_reverse_obligation" in row
        assert "first_mismatch" in row["audit"]

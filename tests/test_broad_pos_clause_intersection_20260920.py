import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "tools" / "bench"))
import broad_pos_clause_intersection as broad  # noqa: E402


def test_complete_left_frontier_continues_into_right_clause(monkeypatch):
    """The live recursion must not return when only the left grammar ends."""
    tiny = {
        "DET": ["an", "some"],
        "NOUN": ["aide", "nine", "memos", "men"],
        "VERB": ["rips", "inspire"],
        "NUM": ["nine"],
        "NAME": ["diana"],
    }
    monkeypatch.setattr(broad, "BANK", tiny)
    rows, nodes = broad.search(
        ("DET", "NOUN", "VERB", "NUM", "NOUN"),
        ("DET", "NOUN", "VERB", "NAME"),
        cap=10,
    )
    assert nodes > 0
    assert (['an', 'aide', 'rips', 'nine', 'memos'],
            ['some', 'men', 'inspire', 'diana']) in rows


def test_exact_audit_is_independent_of_rendering():
    text = "an aide rips nine memos; some men inspire diana"
    checked = broad.audit(text)
    assert checked["exact"]
    assert checked["sha256_forward"] == checked["sha256_reverse"]

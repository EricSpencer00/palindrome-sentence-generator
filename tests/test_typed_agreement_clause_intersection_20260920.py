import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "tools" / "bench"))
from typed_agreement_clause_intersection_20260920 import agreement_ok  # noqa: E402


def test_agreement_gate_keeps_anchor_clauses_and_rejects_malformed_rows():
    assert agreement_ok(
        ["an", "aide", "rips", "nine", "memos"],
        ("DET", "NOUN", "VERB", "NUM", "NOUN"),
    )
    assert agreement_ok(
        ["some", "men", "inspire", "diana"],
        ("DET", "NOUN", "VERB", "NAME"),
    )
    assert not agreement_ok(
        ["a", "late", "men", "inspire", "diana"],
        ("DET", "ADJ", "NOUN", "VERB", "NAME"),
    )

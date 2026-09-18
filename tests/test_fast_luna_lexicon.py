import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.fast_luna_lexicon import (
    SOURCE_CLAUSES,
    _common_lexicon,
    audit_candidate,
    normalize,
    run,
    segment_tape,
)


def test_resegmentation_preserves_the_immutable_tape():
    vocabulary = _common_lexicon(3.0)
    tape = normalize("a dog was in it today")[::-1]
    rows = segment_tape(tape, vocabulary)
    assert rows
    assert all(normalize(row) == tape for row in rows)


def test_independent_audit_reports_an_exact_rendered_surface_and_failures():
    result = run(limit_per_source=8)
    rendered = [row for row in result["records"] if row.get("rendered")]
    assert rendered
    assert all(row["independent_exact_audit"] for row in rendered)
    assert all(row["normalized"] == row["normalized"][::-1] for row in rendered)
    # The current fast run intentionally demonstrates material failures rather
    # than calling an awkward surface readable.
    assert any("right_sentence_shape" in row["rejection_codes"] for row in rendered)
    assert not result["mechanically_admitted"]


def test_one_letter_near_match_is_not_admitted():
    vocabulary = _common_lexicon(3.0)
    catalogue = set()
    exact = audit_candidate("step on", "no pets", catalogue, vocabulary)
    near = audit_candidate("step on", "no pet", catalogue, vocabulary)
    assert exact["checks"]["exact_letter_palindrome"]
    assert not near["checks"]["exact_letter_palindrome"]


def test_sources_are_frozen_and_provenance_is_recorded():
    result = run(limit_per_source=2)
    assert len(result["source_records"]) == len(SOURCE_CLAUSES)
    assert len(result["vocabulary_sha256"]) == 64
    assert result["reader_gate"].startswith("Any future survivor")
    assert result["next_reader_facing_test"]["design"] == "blind pairwise screen"

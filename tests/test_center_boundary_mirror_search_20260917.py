import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.center_boundary_mirror_search_20260917 import (
    _independent_two_pointer,
    audit_rendered,
    run,
)


def test_independent_audits_agree_on_exactness():
    result = run(bound=32)
    assert result["candidate_count"] == 32
    assert result["rendered_rows"]
    for row in result["rendered_rows"]:
        assert row["independent_two_pointer_exact"]
        assert row["independent_sha_exact"]
        assert row["normalized"] == row["normalized"][::-1]
        assert len(row["sha256"]) == 64


def test_center_boundary_and_proper_span_are_distinct_checks():
    row = audit_rendered("ab", "ba", catalogue=set(), lexicon={"ab", "ba"})
    assert row["checks"]["exact_letter_palindrome"]
    assert not row["checks"]["center_inside_word"]
    assert row["checks"]["no_proper_multiword_palindromic_span"]


def test_two_pointer_rejects_near_match():
    assert not _independent_two_pointer("abca")


def test_rows_include_failure_repair_and_provenance():
    result = run(bound=1)
    assert result["failure_repair"]["repair"]
    assert result["rendered_rows"][0]["provenance"].startswith("data/mirror_pairs.json")

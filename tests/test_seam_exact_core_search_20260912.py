from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.seam_exact_core_search_20260912 import (  # noqa: E402
    CORES,
    DEFAULT_BEAM,
    DEFAULT_MAX_DEPTH,
    MAX_LETTERS,
    MIN_LETTERS,
    SEAMS,
    independent_two_pointer,
    run,
    search_core,
    seam_compatible,
    tape,
)


def test_authored_seams_are_length_changing_split_merge_pairs() -> None:
    assert all(seam_compatible(seam) for seam in SEAMS)
    assert all(len(seam.left) == 2 and len(seam.right) == 3 for seam in SEAMS)
    assert len({len(tape(seam.left)) for seam in SEAMS}) > 1


def test_core_search_enforces_exactness_before_retaining_state() -> None:
    result = search_core(CORES[0], max_depth=2, beam=8)
    assert result["exact_closures"]
    assert all(row["construction_exact"] for row in result["exact_closures"])
    assert all(row["independent_two_pointer"]["exact"] for row in result["exact_closures"])
    rejected = [row for row in result["attempts"] if row["hard_rejection"]]
    assert rejected
    assert all(row["rendered"] and row["hard_rejection"] for row in rejected)
    assert any(row["independent_two_pointer"]["exact"] is False for row in rejected)


def test_bounded_run_records_long_exact_closures_and_current_gate() -> None:
    result = run(max_depth=DEFAULT_MAX_DEPTH, beam=DEFAULT_BEAM // 5)
    assert result["config"]["exactness_enforced_during_search"] is True
    assert result["exact_closures"]
    assert max(row["length"] for row in result["exact_closures"]) >= MIN_LETTERS
    assert max(row["length"] for row in result["exact_closures"]) <= MAX_LETTERS
    assert all("current_central_admission" in row for row in result["exact_closures"])
    assert all(row["construction_exact"] for row in result["exact_closures"])
    assert result["mechanically_admitted"] == []
    assert all(not row["intact_grammatical_sentence_witness"]
               and "no_intact_grammatical_sentence_witness" in row["rejection_codes"]
               for row in result["exact_closures"])
    assert all(independent_two_pointer(row["rendered"])["exact"]
               for row in result["exact_closures"])
    assert sum(search["hard_rejections"] for search in result["searches"]) > 0
    assert result["readable_survivors"] == []

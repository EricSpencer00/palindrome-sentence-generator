from __future__ import annotations

from experiments.typed_clause_zipper_20260919 import run


def test_typed_zipper_recovers_independent_anchor_without_finished_reversal():
    result = run()
    rows = result["candidates"]
    assert result["stats"]["exact"] == 2
    assert result["stats"]["longest_exact_letters"] == 38
    anchor = next(row for row in rows if row["rendered"].startswith("An aide"))
    assert anchor["audit"]["two_pointer_exact"]
    assert anchor["audit"]["sha256_forward"] == anchor["audit"]["sha256_reverse"]
    assert anchor["mechanically_admitted"]
    assert anchor["provenance"]["finished_tape_reversed"] is False
    assert anchor["provenance"]["catalogue_imported"] is False

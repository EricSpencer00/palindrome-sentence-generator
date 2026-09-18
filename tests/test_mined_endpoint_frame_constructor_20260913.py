from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.mined_endpoint_frame_constructor_20260913 import (
    FRAME,
    MIN_MATCHED_OUTER,
    independent_two_pointer,
    mine_endpoint_pairs,
    prefix_compatible,
    run,
    search_frame,
)


def test_mined_endpoints_have_substantial_outer_match_and_provenance() -> None:
    rows = mine_endpoint_pairs(limit=3)
    assert rows
    assert all(row["matched_outer_letters"] >= MIN_MATCHED_OUTER for row in rows)
    assert all(row["opening_provenance"]["attested"] for row in rows)
    assert all(row["terminal_provenance"]["attested"] for row in rows)
    assert all(row["opening_provenance"]["source"] == "data/ngrams_wikitext2.json" for row in rows)
    assert all(row["catalogue_source_excluded"] for row in rows)


def test_long_endpoint_pair_is_compatible_before_frame_search() -> None:
    assert prefix_compatible("set a record", "geological survey operates")
    assert not prefix_compatible("the", "letter")


def test_search_retains_rendered_boundary_rejections() -> None:
    endpoint = mine_endpoint_pairs(limit=1)[0]
    result = search_frame(FRAME, endpoint, state_cap=1_000)
    assert result["rows"]
    assert result["stats"]["full_exact_rejections"]
    assert all(row["rendered"].endswith(".") for row in result["rows"])


def test_bounded_run_audits_every_full_frame_and_keeps_gate_closed() -> None:
    result = run(state_cap=1_000, limit=3)
    assert result["config"]["min_matched_outer_letters"] == 6
    assert result["config"]["exact_closure_checked_during_search"]
    assert not result["config"]["catalogue_used_for_generation"]
    assert result["records"]
    assert result["mechanically_admitted"] == []
    for row in result["records"]:
        audit = row["audit"]
        assert audit["sentence_witness"]["independent_surface_parse"]
        assert audit["current_central_admission"]
        assert audit["rejection_codes"]
        assert not audit["mechanically_admitted"]


def test_independent_audit_is_not_constructor_state() -> None:
    assert independent_two_pointer("Ab cdcba.")["exact"]
    assert not independent_two_pointer("Set a record.")["exact"]

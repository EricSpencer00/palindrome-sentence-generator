import json
from pathlib import Path

from experiments.incumbent_568_internal_seam_growth_20260922 import (
    FRONTIER,
    OUT,
    PARENT_SHA256,
    build_payload,
    independent_audit,
    normalize,
)


def test_internal_seam_produces_independently_exact_child():
    payload = build_payload()
    row = payload["rows"][0]
    assert payload["working_incumbent"]["letters"] == 568
    assert payload["working_incumbent"]["sha256"] == PARENT_SHA256
    assert payload["preserved_frontier"] == list(FRONTIER)
    assert payload["stats"]["children_longer_than_568"] == 1
    assert row["id"] == "internal-mara-stops-602"
    assert row["audit"]["letters"] == 602
    assert row["independent_audit"]["two_pointer_exact"]
    assert row["independent_audit"]["sha_equal"]
    assert row["live_seam"]["final_residual"] == ""
    assert "right_cursor_raw_inclusive" not in row["live_seam"]
    assert row["live_seam"]["right_cursor_raw_exclusive"] > row["live_seam"]["left_cursor_raw_exclusive"]
    assert row["live_seam"]["committed_character_contradictions"] == 0
    assert row["grammar_debt"]["inherited_proper_spans"]
    assert row["grammar_debt"]["inherited_repeated_scaffolding"]
    assert row["grammar_debt"]["rough_syntax"]


def test_run_artifact_matches_reproducible_payload():
    assert OUT.exists()
    payload = json.loads(OUT.read_text())
    expected = build_payload()
    assert payload == expected
    assert independent_audit(payload["rows"][0]["rendered"])["two_pointer_exact"]
    assert normalize(payload["rows"][0]["rendered"]) == normalize(
        payload["rows"][0]["rendered"]
    )[::-1]

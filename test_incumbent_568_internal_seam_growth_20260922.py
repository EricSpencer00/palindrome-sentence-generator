import json
from pathlib import Path

from experiments.incumbent_568_internal_seam_growth_20260922 import (
    OUT,
    PARENT_SHA256,
    build_payload,
    normalize,
)


def test_internal_seam_produces_independently_exact_child():
    payload = build_payload()
    row = payload["rows"][0]
    assert payload["working_incumbent"]["letters"] == 568
    assert payload["working_incumbent"]["sha256"] == PARENT_SHA256
    assert payload["stats"]["children_longer_than_568"] == 1
    assert row["audit"]["letters"] == 602
    assert row["independent_audit"]["two_pointer_exact"]
    assert row["independent_audit"]["sha_equal"]
    assert row["live_seam"]["final_residual"] == ""
    assert row["live_seam"]["committed_character_contradictions"] == 0
    assert row["grammar_debt"]["inherited_proper_spans"]
    assert row["grammar_debt"]["inherited_repeated_scaffolding"]
    assert row["grammar_debt"]["rough_syntax"]


def test_run_artifact_matches_reproducible_payload():
    assert OUT.exists()
    payload = json.loads(OUT.read_text())
    expected = build_payload()
    assert payload == expected
    assert normalize(payload["rows"][0]["rendered"]) == normalize(
        payload["rows"][0]["rendered"]
    )[::-1]

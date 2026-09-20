import json

from experiments.live_buffer_grammar_search_20260919 import (
    OUT,
    build_templates,
    independent_audit,
    run,
    search,
)


def test_independent_audit_checks_two_pointers_and_hashes():
    checked = independent_audit("An aide rips nine memos; some men inspire Diana.")
    assert checked["exact"] is True
    assert checked["sha_equal"] is True
    assert checked["letters"] == 38


def test_live_buffers_recover_seed_across_diana_inspire_boundary():
    result = search(build_templates()["seed_control"], limit=4)
    rows = result["candidates"]
    assert any(row["rendered"] == "an aide rips nine memos some men inspire diana" for row in rows)
    assert result["stats"]["exact"] >= 1


def test_run_is_fail_closed_for_reader_claims_and_serializes():
    result = run()
    assert result["provenance"]["post_hoc_repair"] is False
    assert result["provenance"]["finished_tape_reversal"] is False
    assert result["reader_gate"].startswith("closed")
    assert result["exact_candidates"]
    OUT.write_text(json.dumps(result, indent=2) + "\n")

import importlib.util
import sys
from pathlib import Path
P=Path(__file__).parents[1]/"experiments/live_buffer_optional_grammar_search_20260920.py"
spec=importlib.util.spec_from_file_location("fresh",P); fresh=importlib.util.module_from_spec(spec); sys.modules["fresh"]=fresh; spec.loader.exec_module(fresh)
def test_twenty_natural_controls_are_unique_and_not_palindromes():
    assert len(fresh.CONTROLS)==20 and len(set(fresh.CONTROLS))==20
    assert all(not fresh.audit(s)["exact"] for s in fresh.CONTROLS)
def test_search_has_optional_complete_frames_and_live_buffer_audit():
    result=fresh.run(); assert set(result["searches"])=={"base","pp","relative","pp_relative_adjunct"}; assert result["control_count"]==20
    for row in result["exact_candidates"]:
        assert not row["provenance"]["finished_tape_reversal"]; assert not row["provenance"]["post_hoc_repair"]; assert not row["provenance"]["catalogue_text"]
def test_audit_independently_checks_forward_and_reverse_hashes():
    checked=fresh.audit("an aide rips nine memos some men inspire diana"); assert checked["exact"]; assert checked["sha256_forward"]==checked["sha256_reverse"]

import json
from pathlib import Path
from experiments.online_grammar_state_char_decoder_20260916 import audit, novelty_preflight, run, EXPERIMENT_ID

def test_independent_audit_uses_two_pointer_and_hash():
    row = audit("A man, a plan, a canal: Panama")
    assert row["exact"] and row["two_pointer"]["exact"] and row["hash_replay"]

def test_preflight_is_before_generation_and_not_duplicate():
    p = novelty_preflight()
    assert p["performed_before_generation"] and not p["blocked"]

def test_probe_records_actual_prose_and_repair():
    payload = run()
    assert payload["experiment_id"] == EXPERIMENT_ID
    assert payload["stats"]["rendered_probes"] == 3
    assert payload["stats"]["exact"] == 0
    assert all(r["provenance"]["word_order_generated"] for r in payload["rendered_candidates"])
    assert all(r["next_repair"] for r in payload["rendered_candidates"])

import json
from pathlib import Path
from experiments.cfg_cycle_constructor_20260917 import run, audit

def test_recursive_cfg_emits_complete_derivations_and_independent_audit():
    r=run()
    assert r["derivations_checked"] > 1000
    assert r["grammar"] and r["recursive_rule"] == "S -> CLAUSE AND S"
    assert r["exact_count"] == 0
    assert r["shortcut_rejections"]["posthoc_reverse"]
    assert all(x["reader_eligible"] is False for x in r["rendered_frontier"])

def test_audit_rejects_nonpalindrome_and_accepts_known_control():
    assert not audit("quietriver") ["exact_two_pointer"]
    a=audit("amanaplanacanalpanama")
    assert a["exact_two_pointer"] and a["exact_sha256_reverse"]

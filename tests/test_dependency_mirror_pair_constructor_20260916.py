import hashlib, json, re
from pathlib import Path

ROOT = Path(__file__).parents[1]

def letters(s):
    return "".join(re.findall(r"[a-z]", s.lower()))

def test_dependency_pair_keeps_complete_long_prose_and_independent_audits():
    run = json.loads((ROOT / "runs/dependency-mirror-pair-constructor-20260916.json").read_text())
    assert run["novelty_preflight"]["passed"]
    assert run["closure_count"] == 0
    assert len(run["candidates"]) == 4
    best = run["best_actual_prose"]
    assert best["letters"] >= 60
    assert best["rendered"].endswith(".")
    assert best["audit"]["left_two_pointer"]["exact"] is False
    tape = letters(best["rendered"])
    assert best["letters"] == len(tape)
    assert hashlib.sha256(tape.encode()).hexdigest() != hashlib.sha256(tape[::-1].encode()).hexdigest()
    assert best["dependency_provenance"]["left"]["argument_order"] == ["subject", "verb", "object", "adjunct"]
    assert best["provenance"]["catalogue_text_used"] is False
    assert run["repair"][0]["provenance"]["repair"]["operator"] == "heldout-nsubj-and-adjunct-relexicalization"

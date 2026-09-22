import hashlib
import importlib.util
from pathlib import Path

P = Path(__file__).parents[1] / "experiments/abba_clause_live_search_20260926.py"
spec = importlib.util.spec_from_file_location("abba_live", P)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

def test_audit_is_independent_two_pointer_and_sha():
    text = "A man, a plan, a canal: Panama."
    got = mod.audit(text)
    tape = mod.letters(text)
    assert got["two_pointer_exact"]
    assert got["letters"] == len(tape)
    assert got["sha256_forward"] == hashlib.sha256(tape.encode()).hexdigest()

def test_lane_records_residual_repair_and_no_finished_tape_reverse():
    result = mod.run()
    assert result["residual_certificates"]
    assert result["novelty_preflight"]["finished_tape_reversal"] is False
    assert result["provenance"]["reader_gate"].startswith("closed")

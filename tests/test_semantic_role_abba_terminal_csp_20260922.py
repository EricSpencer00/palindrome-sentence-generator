import importlib.util
import sys
from pathlib import Path
spec = importlib.util.spec_from_file_location("m", Path(__file__).parents[1]/"experiments/semantic_role_abba_terminal_csp_20260922.py")
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)

def test_audit_is_independent_two_pointer_and_hash():
    row = m.audit("A man, a plan, a canal: Panama")
    assert row["two_pointer_exact"] and row["sha_exact"]

def test_role_csp_records_outer_residual():
    m.main(); data = m.json.loads(m.RUN.read_text())
    assert data["outer_domain"]["joint_pairs"] == 0
    assert data["inner_combinations"] == 0
    assert data["exact_count"] == 0
    assert data["next_repair"]

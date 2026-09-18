import importlib.util
from pathlib import Path
import sys
import pytest

PATH=Path(__file__).resolve().parents[1]/"experiments/causal_subject_witness_capture_20260913.py"
SPEC=importlib.util.spec_from_file_location("causal_capture_test",PATH)
M=importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name]=M
SPEC.loader.exec_module(M)


def test_captured_state_replays_from_initial_grammar_root():
    grammar=M.CAUSAL.Grammar()
    result=M.solve(grammar,100)
    witness=result["deepest_actual_search_witness"]
    restored=M.replay(grammar,witness["ledger"])
    assert M.digest_state(restored)==witness["state_sha256"]
    assert restored.length==witness["emitted_letters"]
    assert witness["replay_verified"] and not witness["candidate"]


def test_tampered_emission_cannot_replay():
    grammar=M.CAUSAL.Grammar()
    result=M.solve(grammar,100)
    ledger=[dict(action) for action in result["deepest_actual_search_witness"]["ledger"]]
    emission=next(action for action in ledger if action["action"]=="emit")
    emission["character"]="z" if emission["character"]!="z" else "x"
    with pytest.raises(ValueError,match="emission differs"):
        M.replay(grammar,ledger)


def test_interior_expansion_ledger_is_rejected():
    grammar=M.CAUSAL.Grammar()
    result=M.solve(grammar,100)
    ledger=[dict(action) for action in result["deepest_actual_search_witness"]["ledger"]]
    first=next(action for action in ledger if action["action"]=="expand")
    first["production"]="invented"
    with pytest.raises(ValueError,match="unavailable"):
        M.replay(grammar,ledger)

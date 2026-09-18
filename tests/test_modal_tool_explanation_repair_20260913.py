import importlib.util
from pathlib import Path
import sys

PATH = Path(__file__).resolve().parents[1] / "experiments/modal_tool_explanation_repair_20260913.py"
SPEC = importlib.util.spec_from_file_location("modal_instruction_test", PATH)
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)


def test_modal_clause_has_human_subject_and_physical_tool_object():
    grammar = M.Grammar()
    assert M.BASE.parse_tree(grammar, M.CONTROL) is not None
    prod = grammar.productions(M.BASE.sym("CONTENT"))[0]
    assert prod.rhs[0].feature("role") == "craftsperson"
    assert prod.rhs[-1].feature("role") == "tool"
    assert [child.feature("category") for child in prod.rhs[1:-1]] == ["modal", "own"]
    assert all(child.name != "T" for child in prod.rhs)


def test_real_modal_search_crosses_prior_frontier_and_captures_conflict():
    result = M.run()
    witness = result["deepest_actual_search_witness"]
    assert witness["emitted_letters"] > 15
    state = M.CAPTURE.replay(M.Grammar(), witness["ledger"])
    assert state.length == witness["emitted_letters"]
    assert witness["next_character_conflict"]["immediate_emitter_rejects"]
    assert result["exact_closures"] == result["mechanically_admitted_closures"] == []
    assert "no_self_palindromic_proper_multiword_span" in result["control"]["mechanical_checks"]

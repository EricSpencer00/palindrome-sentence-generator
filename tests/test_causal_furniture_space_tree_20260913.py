import importlib.util
from pathlib import Path
import sys

PATH = Path(__file__).resolve().parents[1] / "experiments/causal_furniture_space_tree_20260913.py"
SPEC = importlib.util.spec_from_file_location("causal_furniture_test", PATH)
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)


def test_endpoint_is_new_unequal_and_furniture_heads_are_disjoint():
    grammar = M.Grammar()
    assert grammar.endpoint_rows[0]["opening"] == "a"
    assert grammar.endpoint_rows[0]["ending"] == "sofa"
    assert grammar.endpoint_rows[0]["retained"]
    assert not set(M.LEXICON["moved_furniture"]) & set(M.LEXICON["incoming_furniture"])
    assert M.BASE.parse_tree(grammar, M.CONTROL.replace("heavy table", "heavy sofa")) is None


def test_control_has_complete_causal_space_relation():
    grammar = M.Grammar()
    witness = M.semantic_witness(grammar, M.CONTROL)
    assert witness["independent_complete_reparse"] and witness["distinct_furniture_heads"]
    assert witness["moved_items"] == ["table"] and witness["incoming_items"] == ["sofa"]
    assert M.BASE.exact_audit(M.CONTROL)["letters"] > 100
    root = grammar.productions(grammar.start())[0]
    assert all(child.name != "T" for child in root.rhs)
    assert [child.feature("category") for child in root.rhs[6:9]] == ["need", "space", "for"]


def test_full_search_and_actual_rejection_replay():
    result = M.run()
    witness = result["deepest_actual_search_witness"]
    assert M.CAPTURE.replay(M.Grammar(), witness["ledger"]).length == 13
    assert result["stats"]["states"] == 123 and result["states_exhausted"]
    conflict = witness["next_character_conflict"]
    assert (conflict["debt_source_word"], conflict["expected_character"], conflict["opposing_word"], conflict["opposing_character"]) == ("foster", "r", "velvet", "v")
    assert result["exact_closures"] == result["mechanically_admitted_closures"] == []
    assert "no_self_palindromic_proper_multiword_span" in result["control"]["mechanical_checks"]


def test_repeated_content_is_rejected_by_shared_admission_not_just_example():
    repeated = M.CONTROL.replace("a foster parent", "a carpenter")
    assert M.BASE.parse_tree(M.Grammar(), repeated) is not None
    audit = M.PARENT.audit(M.Grammar(), repeated, "deliberately_repeated_diagnostic")
    assert not audit["mechanical_checks"]["distinct_words"]
    assert not audit["mechanically_admitted"]

import importlib.util
from pathlib import Path
import sys

PATH = Path(__file__).resolve().parents[1] / "experiments/measurement_report_revision_tree_20260913.py"
SPEC = importlib.util.spec_from_file_location("measurement_revision_test", PATH)
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)


def test_comparison_is_object_of_measured_in_complete_causal_clause():
    witness = M.semantic_witness(M.Grammar(), M.CONTROL)
    assert witness["independent_complete_reparse"]
    assert witness["comparison_is_measurement_object"] and witness["fruit_is_measured_source"]
    assert witness["quantity_children"] == ["degree", "nutrient", "than", "expected"]


def test_distinct_heads_and_deferred_words_are_grammar_invariants():
    grammar = M.Grammar()
    roles = [set(M.LEXICON[key]) for key in ("analyst", "measurement_agent", "document", "nutrient")]
    roles.append({word for record in M.FRUITS.values() for word in record["head"]})
    assert all(not left & right for index, left in enumerate(roles) for right in roles[index + 1:])
    pending, seen = [grammar.start()], set()
    while pending:
        lhs = pending.pop()
        if lhs in seen: continue
        seen.add(lhs)
        for production in grammar.productions(lhs):
            if lhs.name != "W": assert all(child.name != "T" for child in production.rhs)
            pending.extend(child for child in production.rhs if child.name != "T")


def test_actual_replay_and_central_gates_are_preserved():
    result = M.run()
    witness = result["deepest_actual_search_witness"]
    assert M.CAPTURE.replay(M.Grammar(), witness["ledger"]).length == witness["emitted_letters"]
    assert witness["next_character_conflict"]["immediate_emitter_rejects"]
    assert result["exact_closures"] == result["mechanically_admitted_closures"] == []
    checks = result["control"]["mechanical_checks"]
    assert checks["distinct_words"] and checks["no_self_palindromic_proper_multiword_span"]
    assert not checks["exact_letter_palindrome"]
    assert result["control"]["independent_exact_audit"]["letters"] > 100

import importlib.util
from pathlib import Path
import sys

PATH = Path(__file__).resolve().parents[1] / "experiments/causal_shutter_reply_tree_20260913.py"
SPEC = importlib.util.spec_from_file_location("shutter_reply_test", PATH)
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)


def test_new_endpoints_retain_overlap_but_reject_word_aligned_interior():
    assert all(row["retained"] for row in M.endpoint_preflight())
    assert all(len(row["opening"]) != len(row["ending"]) for row in M.endpoint_preflight())
    assert not M.endpoint_preflight(("reward",), ("drawer",))[0]["retained"]


def test_reply_reparses_with_question_and_causally_consistent_roles():
    grammar = M.Grammar()
    witness = M.semantic_witness(grammar, M.CONTROL)
    assert witness["independent_complete_reparse"]
    assert witness["context_question"] == "Is the shutter closed?"
    assert witness["state_and_result_share_semantic_mode"]
    assert witness["causal_roles"] == {"agent": "AGENT", "patient": "SHUTTER", "result": "RESULT"}
    assert M.BASE.parse_tree(grammar, M.CONTROL.replace("raised position", "lowered position")) is None


def test_all_new_tree_words_remain_exposed_lexical_slots():
    grammar = M.Grammar()
    seen, pending, forms = set(), [grammar.start()], set()
    while pending:
        lhs = pending.pop()
        if lhs in seen: continue
        seen.add(lhs)
        for production in grammar.productions(lhs):
            if lhs.name != "W": assert all(child.name != "T" for child in production.rhs)
            for child in production.rhs:
                if child.name == "T": forms.add(child.feature("form"))
                else: pending.append(child)
    assert "tell" not in forms and "mallet" not in forms


def test_actual_full_search_ledger_and_lexical_failure_are_preserved():
    result = M.run()
    witness = result["deepest_actual_search_witness"]
    assert M.CAPTURE.replay(M.Grammar(), witness["ledger"]).length == 17
    assert witness["ledger"][0]["production"] == "reply:accessible"
    assert result["stats"]["states"] == 64 and result["states_exhausted"]
    assert result["rejection_certificate"]["all_alternatives_reject"]
    assert result["exact_closures"] == result["mechanically_admitted_closures"] == []
    assert "no_self_palindromic_proper_multiword_span" in result["control"]["mechanical_checks"]

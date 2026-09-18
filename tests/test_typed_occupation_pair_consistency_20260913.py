import importlib.util
from collections import Counter
from pathlib import Path
import sys

PATH = Path(__file__).resolve().parents[1]/"experiments/typed_occupation_pair_consistency_20260913.py"
SPEC = importlib.util.spec_from_file_location("occupation_pair_test", PATH)
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)
B = M.BASE


def test_actual_teacher_met_conflict_and_compatible_occupation_replacement():
    assert not M.overlap_compatible("teacher", "met")
    assert M.overlap_compatible("temp", "met")


def test_full_control_trace_crosses_nineteen_emitted_letters():
    grammar = M.Grammar()
    trace = M.emitted_control_trace(grammar, M.CONTROL)
    assert trace["emitted_letters"] >= 20
    assert trace["matched_pairs"] >= 10
    assert len(trace["emissions"]) == trace["emitted_letters"]
    for event in trace["emissions"]:
        if event["debt_before"]:
            assert event["character"] == event["debt_before"][0]
    assert not trace["complete_palindrome"]
    audit = M.PARENT.audit(grammar, M.CONTROL, "control")
    assert audit["independent_parse"] and audit["independent_exact_audit"]["letters"] > 100


def test_joint_pair_rejects_teacher_met_before_any_emission():
    class PairGrammar(M.Grammar):
        def productions(self, lhs):
            if lhs.name == "W":
                category = lhs.feature("category")
                words = ("teacher", "temp") if category == "person" else ("met",)
                return tuple(B.Production(word, lhs, (B.sym("T", form=word, label=category),)) for word in words)
            return ()
    state = B.State((0,1), (B.Node(0,M.PARENT.word("person")),B.Node(1,M.PARENT.word("past_person"))), (), "",0,0,())
    options = M.successors(PairGrammar(),state,Counter())
    assert len(options) == 1
    assert [leaf.word for leaf in options[0].leaves] == ["temp","met"]
    assert options[0].length == 0


def test_opposite_structural_exposure_keeps_words_unassigned():
    grammar = M.Grammar()
    state = B.State((0,1), (B.Node(0,M.PARENT.word("person")),B.Node(1,M.PARENT.np("object",False))), (), "",0,0,())
    options = M.successors(grammar,state,Counter())
    assert options and all(not option.leaves for option in options)

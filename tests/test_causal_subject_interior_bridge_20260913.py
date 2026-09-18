import importlib.util
from collections import Counter
from pathlib import Path
import sys

PATH=Path(__file__).resolve().parents[1]/"experiments/causal_subject_interior_bridge_20260913.py"
SPEC=importlib.util.spec_from_file_location("causal_subject_test",PATH)
M=importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name]=M
SPEC.loader.exec_module(M)
B=M.BASE


def test_emitted_trace_crosses_twenty_one_before_run():
    grammar=M.Grammar()
    trace=M.PAIR.emitted_control_trace(grammar,M.CONTROL)
    assert trace["emitted_letters"]>21
    assert trace["matched_pairs"]>=13
    assert not trace["complete_palindrome"]
    for event in trace["emissions"]:
        if event["debt_before"]:
            assert event["character"]==event["debt_before"][0]
    row=M.PARENT.audit(grammar,M.CONTROL,"control")
    assert row["independent_parse"] and row["independent_exact_audit"]["letters"]>100
    assert not row["independent_exact_audit"]["exact"]


def test_causal_clause_has_real_subject_and_object():
    witness=M.causal_witness(M.Grammar(),M.CONTROL)
    assert witness["typed_valency_ok"]
    assert len(witness["causal_clauses"])==1
    assert B.parse_tree(M.Grammar(),"reward a temp with a medal because met a drawer") is None
    assert B.parse_tree(M.Grammar(),"reward a temp with a medal because a bishop met a drawer") is not None


def test_causal_structure_does_not_preassign_its_subject():
    grammar=M.Grammar()
    state=B.State((0,),(B.Node(0,B.sym("CAUSE")),),(),"p",1,1,())
    choices=M.PAIR.successors(grammar,state,Counter())
    assert choices
    assert all(not item.leaves for item in choices)
    assert all(item.residual=="p" for item in choices)


def test_workplace_compound_has_unassigned_word_slots():
    grammar=M.Grammar()
    prod=next(p for p in grammar.productions(M.PARENT.np("person",False)) if p.identifier=="np:workplace-person")
    assert all(s.name=="W" and not s.feature("form") for s in prod.rhs)

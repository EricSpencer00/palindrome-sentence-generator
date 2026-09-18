import importlib.util
from pathlib import Path
import sys

PATH=Path(__file__).resolve().parents[1]/"experiments/trade_role_artifact_event_bridge_20260913.py"
SPEC=importlib.util.spec_from_file_location("trade_artifact_test",PATH)
M=importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name]=M
SPEC.loader.exec_module(M)


def test_real_emitted_control_crosses_thirty_five():
    grammar=M.Grammar()
    trace=M.PAIR.emitted_control_trace(grammar,M.CONTROL)
    assert trace["emitted_letters"]>35
    assert trace["matched_pairs"]>=18
    for event in trace["emissions"]:
        if event["debt_before"]:assert event["character"]==event["debt_before"][0]
    assert not trace["complete_palindrome"]


def test_control_uses_human_vendor_and_expert_with_physical_drawer():
    grammar=M.Grammar()
    witness=M.semantic_witness(grammar,M.CONTROL)
    assert witness["independent_parse"]
    assert witness["trade_roles"][0]["words"]==["dessert","street","vendor"]
    assert witness["trade_roles"][0]["agent_type"]=="person"
    assert witness["artifact_events"][0]["agent"]==["the","safety","expert"]
    assert witness["artifact_events"][0]["patient"]==["a","drawer"]
    assert witness["correct_artifact_event_types"]
    assert witness["bare_help_and_drawer_excluded_from_person_lexicon"]
    assert [row["predicate"] for row in witness["rewarded_repairs"]]==[["repairing"],["restoring"]]
    assert all(row["patient_type"]=="object" for row in witness["rewarded_repairs"])


def test_role_construction_does_not_preselect_commodity_or_occupation():
    production=M.Grammar().productions(M.BASE.sym("TRADE",agent_type="person",goods_type="commodity"))[0]
    assert all(symbol.name=="W" and not symbol.feature("form") for symbol in production.rhs)


def test_person_drawer_and_bare_help_cannot_parse():
    grammar=M.Grammar()
    assert M.BASE.parse_tree(grammar,"reward the help with a medal") is None
    assert M.BASE.parse_tree(grammar,"reward a drawer with a medal") is None
    assert M.BASE.parse_tree(grammar,"reward a vendor with a medal after the expert stressed a drawer") is None
    # The vendor head is licensed through its full compositional trade role.
    assert M.BASE.parse_tree(grammar,"reward a dessert street vendor with a medal after the expert stressed a drawer") is not None

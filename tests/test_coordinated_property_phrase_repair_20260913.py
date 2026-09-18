import importlib.util
from pathlib import Path
import sys

PATH=Path(__file__).resolve().parents[1]/"experiments/coordinated_property_phrase_repair_20260913.py"
SPEC=importlib.util.spec_from_file_location("property_phrase_test",PATH)
M=importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name]=M
SPEC.loader.exec_module(M)


def test_emitted_control_crosses_recorded_thirty_three_letter_boundary():
    grammar=M.Grammar()
    trace=M.PAIR.emitted_control_trace(grammar,M.CONTROL)
    assert trace["emitted_letters"]>33
    assert trace["matched_pairs"]>=17
    for event in trace["emissions"]:
        if event["debt_before"]:
            assert event["character"]==event["debt_before"][0]
    assert not trace["complete_palindrome"]


def test_control_contains_complete_independently_parsed_property_phrase():
    grammar=M.Grammar()
    witness=M.property_witness(grammar,M.CONTROL)
    assert witness["independent_parse"]
    assert len(witness["property_phrases"])==1
    assert witness["property_phrases"][0]["complete_typed_phrase"]
    assert M.BASE.parse_tree(grammar,"reward the gentle but help with a medal") is None


def test_property_phrase_leaves_remain_unassigned():
    prod=M.Grammar().productions(M.BASE.sym("PROPERTIES",type="person"))[0]
    assert all(symbol.name=="W" and not symbol.feature("form") for symbol in prod.rhs)


def test_actual_recorded_prior_failure_replays():
    import json
    record=json.loads(M.PRIOR.read_text())["deepest_actual_search_witness"]
    restored=M.CAPTURE.replay(M.CAUSAL.Grammar(),record["ledger"])
    assert M.CAPTURE.digest_state(restored)==record["state_sha256"]
    assert record["next_character_conflict"]["debt_source_word"]=="helper"
    assert record["next_character_conflict"]["opposing_word"]=="gentle"

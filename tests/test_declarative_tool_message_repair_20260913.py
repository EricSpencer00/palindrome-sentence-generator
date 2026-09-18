import importlib.util
from pathlib import Path
import sys

PATH = Path(__file__).resolve().parents[1] / "experiments/declarative_tool_message_repair_20260913.py"
SPEC = importlib.util.spec_from_file_location("declarative_instruction_test", PATH)
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)


def test_control_is_complete_declarative_complement_not_removed_wh_fragment():
    grammar = M.Grammar()
    tree = M.BASE.parse_tree(grammar, M.CONTROL)
    assert tree is not None
    assert tree.production == "instruction:declarative_message"
    message = grammar.productions(M.BASE.sym("MESSAGE"))[0]
    assert message.rhs[0].name == "CRAFT_GROUP"
    assert message.rhs[1].feature("category") == "modal"
    assert message.rhs[-1].feature("role") == "tool"


def test_real_declarative_search_crosses_old_conflict_with_exact_replay():
    result = M.run()
    witness = result["deepest_actual_search_witness"]
    assert witness["emitted_letters"] > 19
    state = M.CAPTURE.replay(M.Grammar(), witness["ledger"])
    assert state.length == witness["emitted_letters"]
    assert witness["next_character_conflict"]["immediate_emitter_rejects"]
    assert result["exact_closures"] == result["mechanically_admitted_closures"] == []

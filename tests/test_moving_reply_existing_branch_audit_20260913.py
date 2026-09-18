import importlib.util
from pathlib import Path
import sys

PATH = Path(__file__).resolve().parents[1] / "experiments/moving_reply_existing_branch_audit_20260913.py"
SPEC = importlib.util.spec_from_file_location("moving_existing_test", PATH)
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)


def test_mode_is_isolated_without_claiming_a_new_construction():
    grammar = M.Grammar()
    assert [row.identifier for row in grammar.productions(grammar.start())] == ["reply:moving"]
    assert M.BASE.parse_tree(grammar, M.CONTROL) is not None
    result = M.run()
    assert not result["new_construction"]
    witness = result["deepest_actual_search_witness"]
    assert M.CAPTURE.replay(grammar, witness["ledger"]).length == 9
    assert witness["next_character_conflict"]["expected_character"] == "i"
    assert witness["next_character_conflict"]["opposing_character"] == "o"
    assert result["stats"]["states"] == 16


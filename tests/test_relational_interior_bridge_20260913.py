import importlib.util
from collections import Counter
from pathlib import Path
import sys

PATH = Path(__file__).resolve().parents[1] / "experiments/relational_interior_bridge_20260913.py"
SPEC = importlib.util.spec_from_file_location("bridge_test", PATH)
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)
B = M.BASE


def test_intact_control_has_two_complete_typed_bridges():
    row = M.PARENT.audit(M.BridgeGrammar(), M.CONTROL, "control")
    assert row["independent_parse"]
    assert row["independent_exact_audit"]["letters"] > 100
    assert not row["independent_exact_audit"]["exact"]
    witness = M.bridge_witness(M.BridgeGrammar(), M.CONTROL)
    assert len(witness["bridges"]) == 2
    assert witness["all_complements_complete_and_typed"]


def test_incomplete_or_mistyped_bridge_fails_reparse():
    grammar = M.BridgeGrammar()
    assert B.parse_tree(grammar, "draw the portrait of") is None
    assert B.parse_tree(grammar, "draw the portrait of the garden") is None
    assert B.parse_tree(grammar, "draw the portrait of the artist") is not None


def test_bridge_structure_does_not_select_interior_words():
    grammar = M.BridgeGrammar()
    symbol = B.sym("BRIDGE", remaining="2")
    state = B.State((0,), (B.Node(0, symbol),), (), "a", -1, 1, ())
    options = M.PARENT.successors(grammar, state, Counter())
    assert options
    assert all(not option.leaves for option in options)
    assert all(option.residual == "a" and option.owner == -1 for option in options)


def test_exposed_bridge_word_obeys_existing_character_debt():
    grammar = M.BridgeGrammar()
    symbol = B.sym("W", category="relation:of")
    state = B.State((0,), (B.Node(0, symbol),), (), "x", -1, 1, ())
    assert not M.PARENT.successors(grammar, state, Counter())
    matching = B.State(state.frontier, state.nodes, (), "o", -1, 1, ())
    chosen = M.PARENT.successors(grammar, matching, Counter())[0]
    consumed = M.PARENT.successors(grammar, chosen, Counter())[0]
    assert consumed.residual == ""
    assert consumed.length == 2

import importlib.util
from collections import Counter
from pathlib import Path
import sys

PATH = Path(__file__).resolve().parents[1] / "experiments/shallow_imperative_lockstep_20260913.py"
SPEC = importlib.util.spec_from_file_location("shallow_lockstep_test", PATH)
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)
B = M.BASE


def initial(grammar):
    return B.State((0,), (B.Node(0, grammar.start()),), (), "", 0, 0, ())


def test_control_is_long_intact_independently_parsed_and_diagnostic():
    row = M.audit(M.Grammar(), M.CONTROL, "control")
    assert row["independent_parse"]
    assert row["independent_exact_audit"]["letters"] > 100
    assert not row["independent_exact_audit"]["exact"]
    assert row["grammar_witness"]["max_relative_depth"] == 1
    assert row["grammar_witness"]["valency_ok"]
    assert row["mechanical_checks"]["distinct_words"]


def test_structural_expansion_never_lexicalizes_an_unexposed_word():
    grammar = M.Grammar()
    state = M.successors(grammar, initial(grammar), Counter())[0]
    assert not state.leaves
    state = M.successors(grammar, state, Counter())[0]
    assert not state.leaves
    state = M.successors(grammar, state, Counter())[0]
    assert [leaf.word for leaf in state.leaves] == ["draw"]
    assert all(node.symbol.name != "T" for node in state.nodes if node.identifier in state.frontier[1:])


class ToyGrammar(M.Grammar):
    """Boundary-shifted a/bcba has unmatched word boundaries but exact tape."""
    def productions(self, lhs):
        if lhs.name == "S":
            return (B.Production("root", lhs, (M.word("left"), M.word("right"))),)
        if lhs.name == "W":
            form = "a" if lhs.feature("category") == "left" else "bcba"
            return (B.Production("toy", lhs, (B.sym("T", form=form, label="toy"),)),)
        return ()


def test_canonical_character_schedule_preserves_shifted_boundaries():
    grammar = ToyGrammar()
    state, stats = initial(grammar), Counter()
    while not B.complete(state):
        options = M.successors(grammar, state, stats)
        assert len(options) == 1
        state = options[0]
    assert B.render(state) == "a bcba"
    assert B.exact_audit(B.render(state))["exact"]
    assert state.length == 5
    assert stats["characters_emitted"] == 5


def test_mismatch_is_rejected_at_next_character():
    grammar = ToyGrammar()
    state = initial(grammar)
    while not state.residual:
        state = M.successors(grammar, state, Counter())[0]
    assert state.residual == "a" and state.owner == 1
    assert M.active_edge(state)[0] == -1
    # A wrong active endpoint cannot be consumed by the character checker.
    state = B.State((1,), (B.Node(1, B.sym("T", form="z"), terminal="z"),),
                    (B.Leaf(1, 1, "z", "toy"),), "a", 1, 1, ())
    assert M.successors(grammar, state, Counter()) == ()


def test_bad_article_is_not_admitted():
    assert not M.article_ok("help a eager teacher")
    assert M.article_ok("help the eager teacher")


def test_independent_reparse_rejects_incomplete_relative():
    assert B.parse_tree(M.Grammar(), "help the teacher who helped") is None
    assert B.parse_tree(M.Grammar(), "help the teacher who helped the artist") is not None


def test_root_article_and_following_lexical_seam_is_satisfiable():
    oracle = M.seam_oracle(M.Grammar())
    assert oracle["independent_parse"]
    assert oracle["outer_matching_character_pairs"] >= 7
    assert oracle["matching_prefix"] == oracle["matching_suffix"][::-1]
    assert not oracle["exact"]

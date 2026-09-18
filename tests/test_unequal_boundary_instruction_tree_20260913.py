import importlib.util
from pathlib import Path
import sys

PATH = Path(__file__).resolve().parents[1] / "experiments/unequal_boundary_instruction_tree_20260913.py"
SPEC = importlib.util.spec_from_file_location("unequal_instruction_test", PATH)
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)


def test_reverse_outer_word_gate_rejects_known_bad_topology():
    rejected = M.endpoint_preflight(("reward",), ("drawer",))[0]
    assert rejected["forces_proper_palindromic_interior"]
    assert not rejected["retained"]
    retained = M.endpoint_preflight(("tell",), ("mallet",))[0]
    assert retained["retained"] and len(retained["opening"]) != len(retained["ending"])


def test_control_reparses_as_long_complete_instruction():
    row = M.PARENT.audit(M.Grammar(), M.CONTROL, "diagnostic")
    assert row["independent_parse"]
    assert row["independent_exact_audit"]["letters"] == 125
    assert not row["independent_exact_audit"]["exact"]
    assert "no_self_palindromic_proper_multiword_span" in row["mechanical_checks"]
    assert not row["mechanically_admitted"]


def test_every_nonlexical_production_leaves_words_unassigned():
    grammar = M.Grammar()
    pending, seen = [grammar.start()], set()
    while pending:
        lhs = pending.pop()
        if lhs in seen or lhs.name == "T": continue
        seen.add(lhs)
        for production in grammar.productions(lhs):
            if lhs.name != "W":
                assert all(child.name != "T" for child in production.rhs)
            pending.extend(production.rhs)


def test_actual_first_frontier_has_exhaustive_lexical_certificate():
    result = M.run()
    assert result["stats"]["states"] == 459 and result["states_exhausted"]
    witness = result["deepest_actual_search_witness"]
    assert witness["emitted_letters"] == 15 and witness["replay_verified"]
    assert M.CAPTURE.replay(M.Grammar(), witness["ledger"]).residual == "n"
    certificate = result["lexical_rejection_certificate"]
    assert certificate["all_lexical_alternatives_reject"]
    assert set(row["offered_character"] for row in certificate["alternatives"]) == {"d", "e", "t"}
    assert result["exact_closures"] == result["mechanically_admitted_closures"] == []

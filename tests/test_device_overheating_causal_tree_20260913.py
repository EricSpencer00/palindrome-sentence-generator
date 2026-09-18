import importlib.util
from pathlib import Path
import sys

PATH = Path(__file__).resolve().parents[1] / "experiments/device_overheating_causal_tree_20260913.py"
SPEC = importlib.util.spec_from_file_location("device_overheating_test", PATH)
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)


def test_central_endpoint_provenance_runs_before_expansion():
    grammar = M.Grammar()
    assert grammar.endings == ("demos",)
    assert all(row["central_endpoint_provenance_allowed"] for row in grammar.endpoint_rows)
    assert M.has_forbidden_catalogue_endpoint_scaffold(("no", "it", "is", "open", "position"))
    assert not M.has_forbidden_catalogue_endpoint_scaffold(("some", "devices", "demos"))


def test_control_has_coreferential_cause_and_no_length_padding_relative():
    grammar = M.Grammar()
    witness = M.semantic_witness(grammar, M.CONTROL)
    assert witness["independent_complete_reparse"] and witness["equipment_coreference"]
    assert witness["plural_anaphor"] and witness["lack_of_rest_is_part_of_use_event"]
    assert witness["demonstration_is_use_purpose"]
    assert "who" not in M.CONTROL.split()
    assert M.BASE.parse_tree(grammar, M.CONTROL.replace("operated them", "operated him")) is None


def test_lexical_roles_are_disjoint_and_all_structural_words_unassigned():
    grammar = M.Grammar()
    groups = [set(M.LEXICON[role]) for role in ("device", "operator", "demonstration", "break")]
    assert all(not left & right for index, left in enumerate(groups) for right in groups[index + 1:])
    pending, seen = [grammar.start()], set()
    while pending:
        lhs = pending.pop()
        if lhs in seen: continue
        seen.add(lhs)
        for production in grammar.productions(lhs):
            if lhs.name != "W": assert all(child.name != "T" for child in production.rhs)
            pending.extend(child for child in production.rhs if child.name != "T")


def test_actual_search_replays_and_all_new_central_gates_remain_present():
    result = M.run()
    witness = result["deepest_actual_search_witness"]
    assert M.CAPTURE.replay(M.Grammar(), witness["ledger"]).length == witness["emitted_letters"]
    assert witness["next_character_conflict"]["immediate_emitter_rejects"]
    checks = result["control"]["mechanical_checks"]
    assert checks["not_forbidden_catalogue_endpoint_scaffold"] and checks["no_self_palindromic_proper_multiword_span"]
    assert checks["distinct_words"]
    assert result["exact_closures"] == result["mechanically_admitted_closures"] == []

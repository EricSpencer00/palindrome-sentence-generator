import importlib.util
from pathlib import Path
import sys

PATH = Path(__file__).resolve().parents[1] / "experiments/perfect_craft_report_repair_20260913.py"
SPEC = importlib.util.spec_from_file_location("perfect_craft_test", PATH)
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)


def test_report_is_finite_typed_and_has_no_preassigned_words():
    grammar = M.Grammar()
    for prod in grammar.productions(M.BASE.sym("CRAFT_REPORT")):
        assert prod.rhs[0].feature("category") == "first_person_plural"
        assert prod.rhs[1].feature("category") == "perfect_auxiliary"
        assert prod.rhs[-2].feature("category") == "craft_participle"
        assert prod.rhs[-1].feature("role") == "tool"
        assert all(child.name != "T" for child in prod.rhs)
    assert all(not row["forces_proper_palindromic_interior"] for row in grammar.retained)


def test_control_independently_reparses_as_report_not_ownership_message():
    grammar = M.Grammar()
    tree = M.BASE.parse_tree(grammar, M.CONTROL)
    assert tree is not None and tree.production == "instruction:reported_craft_event"
    witness = M.semantic_witness(grammar, M.CONTROL)
    assert witness["typed_report"] == [{"subject": "first_person_plural", "finite_auxiliary": "perfect_auxiliary",
                                        "predicate": "craft_participle", "patient": "tool"}]


def test_search_persists_actual_new_report_path_and_central_admission():
    result = M.run()
    witness = result["deepest_actual_search_witness"]
    restored = M.CAPTURE.replay(M.Grammar(), witness["ledger"])
    assert restored.length == witness["emitted_letters"]
    assert witness["ledger"][0]["production"] == "instruction:reported_craft_event"
    assert witness["next_character_conflict"]["immediate_emitter_rejects"]
    assert result["exact_closures"] == result["mechanically_admitted_closures"] == []
    assert "no_self_palindromic_proper_multiword_span" in result["control"]["mechanical_checks"]

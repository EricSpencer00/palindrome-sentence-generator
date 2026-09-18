import importlib.util
from pathlib import Path
import sys

PATH=Path(__file__).resolve().parents[1]/"experiments/joint_trade_safety_seam_enumeration_20260913.py"
SPEC=importlib.util.spec_from_file_location("joint_seam_test",PATH)
M=importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name]=M
SPEC.loader.exec_module(M)


def test_exact_recorded_street_expert_boundary():
    match=M.matched_prefix("reward a dessert street vendor","the safety expert stressed a drawer")
    assert match["matched_pairs"]==18
    assert (match["next_left"],match["next_right"])==("e","p")


def test_phrase_lexicons_exclude_person_help_and_drawer():
    assert "help" not in M.SAFETY_MODIFIERS and "drawer" not in M.SAFETY_MODIFIERS
    assert "street" in M.VENUE_HEADS
    assert "shopkeeper" not in M.VENUE_HEADS["street"]


def test_shared_orthography_does_not_imply_a_complete_palindrome():
    match=M.matched_prefix("reward a dessert street vendor","the safety expert stressed a drawer")
    assert match["left_matched"]==match["right_matched"]
    assert not match["fully_consumed_one_phrase"]

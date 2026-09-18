"""Every prospective constructor must expose the shared exclusion result."""
from experiments.character_crossing_relative_clause import audit as relative_audit
from experiments.centre_crossing_sentence_decoder import sentence_checks
from experiments.fast_luna_lexicon import audit_candidate
from experiments.joint_dual_lexicalization import mechanical_checks
from experiments.guided_word_mirror_authoring import screen_left
from experiments.lexicalized_dependency_tree import checks as tree_checks
from experiments.llm_exact_authoring_loop import checks as llm_checks
from experiments.semantic_slot_solver import audit as semantic_audit
from experiments.whole_sentence_shared_tape import checks as whole_checks


CATALOGUE_FRAME = "Marge lets Hara see Sarah's telegram."


class NeverComplete:
    def complete(self, words):
        return False


def test_every_final_constructor_screen_rejects_catalogue_frame():
    assert not relative_audit(CATALOGUE_FRAME, set())["not_catalogue_family_derivative"]
    assert not tree_checks(CATALOGUE_FRAME, set())["not_catalogue_family_derivative"]
    assert not semantic_audit(CATALOGUE_FRAME, set())["not_catalogue_family_derivative"]
    assert not whole_checks(CATALOGUE_FRAME.lower().rstrip(".").split())["not_catalogue_family_derivative"]
    assert not sentence_checks(
        CATALOGUE_FRAME.lower().rstrip(".").split(), NeverComplete()
    )["not_catalogue_family_derivative"]
    assert not mechanical_checks(
        "marge lets hara", "see sarah's telegram", existing_pairs=set()
    )["not_catalogue_family_derivative"]
    assert not audit_candidate(
        "marge lets hara", "see sarah's telegram", set(),
        {"marge", "lets", "hara", "see", "sarahs", "telegram"},
    )["checks"]["not_catalogue_family_derivative"]


def test_mirror_authors_cannot_bypass_the_shared_gate():
    whole_word_mirror = "Rats desserts deliver reviled stressed star."
    assert not llm_checks(whole_word_mirror)["not_word_order_symmetry"]
    row = screen_left(
        "rats desserts deliver",
        {"rats": "star", "desserts": "stressed", "deliver": "reviled"},
        existing_pairs=set(), novel_checker=lambda _: True,
    )
    assert not row["checks"]["not_word_order_symmetry"]

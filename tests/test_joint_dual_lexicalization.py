from experiments.joint_dual_lexicalization import mechanical_checks, pair_identity


def test_pair_identity_ignores_orientation_and_spacing():
    assert pair_identity("step on", "no pets") == pair_identity("no pets", "step on")


def test_mechanical_screen_keeps_all_hard_failures_visible():
    checks = mechanical_checks(
        "step on", "no pet", existing_pairs=set(),
        word_checker=lambda words: True, novel_checker=lambda text: True)
    assert checks["reverse_match"] is False
    assert checks["exact_palindrome"] is False
    assert checks["length_band"] is False


def test_exact_pair_can_still_be_rejected_as_known_material():
    checks = mechanical_checks(
        "step on", "no pets", existing_pairs={pair_identity("step on", "no pets")},
        word_checker=lambda words: True, novel_checker=lambda text: False)
    assert checks["reverse_match"] is True
    assert checks["exact_palindrome"] is True
    assert checks["novel_catalogue"] is False
    assert checks["novel_v3_bank_pair"] is False

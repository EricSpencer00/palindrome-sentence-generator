from experiments.fast_luna_repair import (
    first_mismatch,
    local_repair,
    mismatch_positions,
    mismatch_word_slots,
    reader_test_package,
    screen,
    word_order_mirror,
)


def test_mismatch_report_maps_both_directions_to_word_slots():
    assert first_mismatch("abcdef") == 0
    assert mismatch_positions("abcdef") == (0, 1, 2, 3, 4, 5)
    assert mismatch_word_slots("ab cd ef.") == (0, 1, 2)


def test_local_repair_changes_only_mismatch_touching_slots_and_closes_exactly():
    rows = local_repair("ab cd ef.", {2: ["cdcba"]}, max_mutations=1)
    assert len(rows) == 1
    assert rows[0].text == "Ab cd cdcba."
    assert rows[0].changed_slots == (2,)
    assert rows[0].text.replace(" ", "").replace(".", "").lower() == "abcdcdcba"


def test_screen_rejects_mirror_phrases_and_repeated_or_self_palindromic_units():
    assert word_order_mirror(("step", "on", "no", "pets"))
    checks = screen(
        "Step on no pets.",
        vocabulary={"step", "on", "no", "pets"},
        known=set(), min_letters=4,
    )
    assert checks["exact_letter_palindrome"]
    assert not checks["no_word_order_mirror"]

    repeated = screen("Level level.", vocabulary={"level"}, known=set(), min_letters=1)
    assert not repeated["no_repeated_words"]
    assert not repeated["no_self_palindromic_word_units"]


def test_reader_package_keeps_candidate_and_matched_shuffle_separate():
    package = reader_test_package("Careful editors revise old drafts.")
    assert package["status"] == "not_run"
    assert package["candidate_intact"] == "Careful editors revise old drafts."
    assert package["matched_word_shuffle"] != package["candidate_intact"]

from tools.polaris.sentence_bank_debug import split_pair


def test_split_pair_requires_a_word_boundary_at_letter_midpoint():
    assert split_pair(["rats", "live", "evil", "star"]) == (
        ["rats", "live"], ["evil", "star"])
    assert split_pair(["racecar"]) is None

from experiments.centre_crossing_sentence_decoder import render_crossing


def test_centre_marker_is_merged_into_words_not_shown_as_a_unit():
    rows = render_crossing(("me",), ("tide",), "t")
    assert ["met", "tide"] in rows
    assert all("t" not in row for row in rows)


def test_no_crossing_render_without_words_on_both_sides():
    assert render_crossing((), ("tide",), "t") == []

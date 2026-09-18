from experiments.lexical_bridge_fresh_short_sentences_20260913 import (
    SOURCE_SENTENCES,
    VOCABULARY,
    first_lexical_dead_frontier,
    reverse_tape,
    run,
    segment_reversed_tape,
)


def test_sources_are_complete_short_sentences_before_bridge_search():
    result = run()
    assert len(result["source_records"]) == len(SOURCE_SENTENCES) == 5
    assert all(row["source_complete_sentence"] for row in result["source_records"])
    assert all(30 <= row["source_letters"] <= 60 for row in result["source_records"])
    assert result["config"]["lengthen_only_after_exact_candidate"]
    assert result["config"]["right_boundaries_cross_source_words"]


def test_reversed_segmentation_is_immutable_and_records_first_real_failure():
    result = run(segmentation_limit=256)
    assert result["records"] == []
    assert result["exact_candidates"] == []
    assert result["parsed_exact_candidates"] == []
    assert result["mechanically_admitted"] == []
    failure = result["first_real_bridge_failure"]
    assert failure["source_id"] == "B01"
    assert failure["reason"] == "no_exact_lexical_word_boundary_segmentation"
    assert failure["dead_frontier"]["position"] == 9
    assert failure["dead_frontier"]["prefix"] == "emohtadae"
    assert failure["dead_frontier"]["remaining"].startswith("rb")


def test_segmenter_never_changes_the_reversed_tape():
    for source in SOURCE_SENTENCES:
        tape = reverse_tape(source)
        for words in segment_reversed_tape(tape, VOCABULARY, limit=256):
            assert "".join(words) == tape
            assert reverse_tape(source) == tape
    assert first_lexical_dead_frontier(reverse_tape(SOURCE_SENTENCES[0]), VOCABULARY)["position"] == 9

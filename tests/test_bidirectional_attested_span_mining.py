from experiments.bidirectional_attested_span_mining import (
    attested_spans,
    intersect_reverse_spans,
)


def test_span_intersection_recovers_independently_attested_mirror_readings():
    rows = [
        {"text": "go hang a salami", "letters": "gohangasalami"},
        {"text": "ima lasagna hog", "letters": "imalasagnahog"},
    ]
    stats, pairs = intersect_reverse_spans(rows, novel_checker=lambda text: True)
    assert stats["candidate_pairs"] == 1
    assert pairs == [{
        "left": "go hang a salami",
        "right": "ima lasagna hog",
        "text": "go hang a salami ima lasagna hog",
        "letters_per_side": 13,
        "left_attestations": 1,
        "right_attestations": 1,
        "novel_catalogue": True,
        "exact_palindrome": True,
    }]


def test_span_generator_requires_an_in_sentence_common_word_run():
    spans = list(attested_spans(
        [["the", "quiet", "cat", "zzzx", "slept"]],
        vocab={"the", "quiet", "cat", "slept"}, min_words=2, max_words=3,
        min_letters=6, max_letters=20))
    assert [row["text"] for row in spans] == ["the quiet", "the quiet cat", "quiet cat"]

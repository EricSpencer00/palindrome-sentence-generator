from collections import Counter

from experiments.audit_programmatic_readability import (
    BrownBigramModel,
    matched_control_contrasts,
    order_gain,
    order_gain_by_sentence,
    repeated_bigram_rate,
)


def test_repeated_bigram_rate_counts_only_later_occurrences():
    assert repeated_bigram_rate(["a", "b", "a", "b", "a"]) == 0.5
    assert repeated_bigram_rate(["a"]) == 0.0


def test_order_gain_is_deterministic_for_a_fixed_item_id():
    model = BrownBigramModel(
        Counter({"<s>": 3, "a": 3, "b": 3, "</s>": 3}),
        Counter({("<s>", "a"): 3, ("a", "b"): 3, ("b", "</s>"): 3}),
        vocabulary_size=4,
    )
    first = order_gain(model, ["a", "b"], "R001", 7, 8)
    second = order_gain(model, ["a", "b"], "R001", 7, 8)
    assert first == second
    assert first[1] is not None and first[1] > 0


def test_sentence_aware_score_adds_a_boundary_pair_for_each_sentence():
    model = BrownBigramModel(
        Counter({"<s>": 4, "a": 4, "b": 4, "</s>": 4}),
        Counter({("<s>", "a"): 3, ("a", "b"): 1,
                 ("b", "</s>"): 2, ("a", "</s>"): 2,
                 ("<s>", "b"): 1}),
        vocabulary_size=4,
    )
    one_sentence = model.score_sentences([["a", "b"]])
    two_sentences = model.score_sentences([["a"], ["b"]])
    assert one_sentence != two_sentences
    first = order_gain_by_sentence(model, [["a"], ["b"]], "R002", 7, 8)
    second = order_gain_by_sentence(model, [["a"], ["b"]], "R002", 7, 8)
    assert first == second


def test_matched_control_contrasts_pair_identical_word_multisets():
    rows = [
        {"source": "real_prose_control", "word_multiset_signature": "a\\x00b",
         "brown_bigram_logprob": 3.0, "brown_order_gain_vs_own_shuffle": 2.0,
         "mean_zipf_frequency": 1.0, "repeated_word_rate": 0.0,
         "repeated_bigram_rate": 0.0, "punctuation_segments_per_100_words": 1.0},
        {"source": "shuffled_word_control", "word_multiset_signature": "a\\x00b",
         "brown_bigram_logprob": 1.0, "brown_order_gain_vs_own_shuffle": 0.5,
         "mean_zipf_frequency": 1.0, "repeated_word_rate": 0.0,
         "repeated_bigram_rate": 0.0, "punctuation_segments_per_100_words": 1.0},
    ]
    result = matched_control_contrasts(rows)
    assert result["matched_pairs"] == 1
    assert result["real_prose_minus_shuffled"]["brown_order_gain_vs_own_shuffle"] == 1.5

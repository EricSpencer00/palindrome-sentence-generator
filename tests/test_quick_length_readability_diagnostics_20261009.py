from collections import Counter
import pytest

from experiments.quick_length_readability_diagnostics_20261009 import (
    analyze, features, own_word_shuffles, prose_rows, sentences,
)
from llm_palindrome.bigram import BigramModel


def model():
    return BigramModel({("cats", "eat"): 100, ("eat", "fish"): 100,
                        ("birds", "fly"): 100},
                       {"cats": 100, "eat": 200, "fish": 100, "birds": 100, "fly": 100})


def test_shuffles_preserve_word_multiset_and_each_sentence_length():
    parts = sentences("Cats eat fish. Birds fly!")
    shuffled = list(own_word_shuffles(parts, 32))
    assert shuffled == list(own_word_shuffles(parts, 32))
    assert len(shuffled) == 32
    for item in shuffled:
        assert list(map(len, item)) == [3, 2]
        assert Counter(w for s in item for w in s) == Counter(w for s in parts for w in s)
    result = features("Cats eat fish. Birds fly!", model(), set(), set())
    assert result["scored_pair_count"] == 3  # No fish/birds boundary pair.
    assert result["attested_pair_rate"] == 1
    assert result["local_order_gain_nats_per_pair"] > 0


def test_exactness_lookup_and_repetition_are_separate_from_local_order():
    row = features("Madam? Madam!", model(), {"madammadam"}, set())
    assert row["exact_palindrome"]
    assert row["in_reddit_exact_lookup"]
    assert not row["absent_from_both_lookups"]
    assert row["repeated_word_rate"] == 0.5
    assert row["repeated_bigram_rate"] == 0  # Sentence boundaries respected.
    assert row["local_order_gain_nats_per_pair"] is None
    assert not features("123!?", model(), set(), set())["exact_palindrome"]
    assert not features("éve", model(), set(), set())["exact_palindrome"]


def test_prose_loading_requires_labels_and_preserves_nonpalindromes():
    texts = [{"id": "a", "text": "Cats eat fish."}, {"id": "b", "text": "Fish cats eat."}]
    keys = [{"id": "a", "source": "real_prose_control"}, {"id": "b", "source": "shuffled_word_control"}]
    assert prose_rows(texts, keys) == [{**keys[0], **texts[0]}]
    with pytest.raises(ValueError, match="labelled"):
        prose_rows(texts)


def test_failed_jobs_stay_in_denominator_and_longest_stays_separate():
    base = dict(direction="outside_in", inventory="words", min_letters=1, max_letters=40, seed=1)
    jobs = [{**base, "id": "failed", "status": "hard_deadline", "selected": None,
             "longest_exact": {"text": "Madam."}},
            {**base, "id": "invalid", "status": "completed_with_output",
             "selected": {"text": "Cats eat fish."}, "longest_exact": None},
            {**base, "id": "valid", "status": "completed_with_output",
             "selected": {"text": "Madam."}, "longest_exact": {"text": "Madam."}}]
    credit = {"author": "u/example", "source_url": "https://www.reddit.com/example"}
    report = analyze({"status": "completed_diagnostic", "results": jobs},
                     {"results": [{"id": "cur", "surface": "Madam."}]},
                     {"exact": [{"text": "Madam.", "sources": [credit]}]},
                     [{"id": "prose", "text": "Cats eat fish.", "source": "real_prose_control"}],
                     ["madam"], model())
    summary = next(s for s in report["summaries"] if s["cohort"].startswith("new/") and s["band"] == "1-40")
    assert summary["records"] == 3
    assert summary["selected_outputs"] == 2
    assert summary["exact_outputs"] == 1
    assert summary["non_palindrome_outputs"] == 1
    assert summary["no_selected_output"] == 1
    assert summary["exact_output_rate_per_requested_job"] == 1 / 3
    assert summary["jobs_with_any_exact_closure"] == 2
    prose = next(s for s in report["summaries"] if s["cohort"] == "ordinary_prose_control" and s["band"] == "1-40")
    assert prose["metric_rows"] == 1 and prose["exact_outputs"] == 0
    assert next(r for r in report["rows"] if r["cohort"] == "reddit_credited")["sources"] == [credit]
    assert report["rows"][0]["text"] is None
    assert report["rows"][0]["longest_exact_features"]["exact_palindrome"]

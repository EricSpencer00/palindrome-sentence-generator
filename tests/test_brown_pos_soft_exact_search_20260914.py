from collections import Counter

from experiments.brown_pos_soft_exact_search_20260914 import BrownPOSScorer


def test_pos_scorer_prefers_seen_transition_over_unseen_transition():
    scorer = BrownPOSScorer(
        {"baker": frozenset({"NOUN"}), "shares": frozenset({"VERB"}),
         "bread": frozenset({"NOUN"})},
        Counter({("NOUN", "VERB"): 10, ("VERB", "NOUN"): 8}),
        Counter({"NOUN": 10, "VERB": 8}),
        Counter({"NOUN": 4, "VERB": 2}),
        6,
    )
    # ``beam_search`` passes the post-insertion left tuple to the scorer.
    seen = scorer.word_delta(("baker", "shares", "bread"), (), "L", "bread", "append")
    unseen = scorer.word_delta(("baker", "bread"), (), "L", "bread", "append")
    assert seen > unseen


def test_pos_scorer_has_no_readability_certification_surface():
    assert not hasattr(BrownPOSScorer, "certify_readability")

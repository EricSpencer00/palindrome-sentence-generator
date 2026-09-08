from llm_palindrome.bigram import BigramModel
from llm_palindrome.semantic import RankOrderScorer


def test_order_term_prefers_attested_join_over_unseen_join():
    words = ["the", "cat", "zebra"]
    bg = BigramModel({("the", "cat"): 10}, {"the": 20, "cat": 10, "zebra": 10})
    scorer = RankOrderScorer(words, bg, order_weight=1.0, length_weight=0.0)
    cat = scorer.word_delta(("the", "cat"), (), "L", "cat", "append")
    zebra = scorer.word_delta(("the", "zebra"), (), "L", "zebra", "append")
    assert cat > zebra


def test_zero_order_weight_is_dependency_free_rank_baseline():
    scorer = RankOrderScorer(["the", "cat"], order_weight=0.0)
    assert scorer.word_delta(("the",), (), "L", "the", "append") > 0

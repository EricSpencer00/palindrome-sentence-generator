from llm_palindrome.clause_ngram import ClauseNgramScorer


def test_clause_ngram_rewards_attested_multiword_history_both_directions():
    scorer = ClauseNgramScorer(
        [("the", "dog", "runs"), ("a", "cat", "sleeps")], order=3)
    assert scorer.word_delta((), ("the", "dog", "runs"), "R", "runs", "append") > \
        scorer.word_delta((), ("the", "cat", "runs"), "R", "runs", "append")
    assert scorer.word_delta(("the", "dog", "runs"), (), "L", "the", "prepend") > \
        scorer.word_delta(("the", "cat", "runs"), (), "L", "the", "prepend")


def test_clause_ngram_accepts_a_generator_once():
    scorer = ClauseNgramScorer((row for row in [("we", "can", "go")]))
    assert scorer.vocab == 3

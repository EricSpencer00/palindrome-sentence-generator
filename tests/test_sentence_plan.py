from llm_palindrome.sentence_plan import SentencePlan


TABLE = {
    "the": {"DET"}, "dog": {"NOUN"}, "runs": {"VERB"},
    "quietly": {"ADV"}, "blue": {"ADJ"}, "stone": {"NOUN"},
}
SHAPES = {("DET", "NOUN", "VERB"),
          ("ADJ", "NOUN", "VERB", "ADV")}


def test_sentence_plan_tracks_right_prefix_and_left_suffix():
    plan = SentencePlan(TABLE, SHAPES)
    assert plan.prefix_possible(["the"])
    assert plan.prefix_possible(["the", "dog"])
    assert not plan.prefix_possible(["runs"])
    assert plan.suffix_possible(["runs"])
    assert plan.suffix_possible(["dog", "runs"])
    assert not plan.suffix_possible(["the"])


def test_sentence_plan_requires_one_complete_subject_verb_reading():
    plan = SentencePlan(TABLE, SHAPES)
    assert plan.complete(["the", "dog", "runs"])
    assert plan.complete(["blue", "stone", "runs", "quietly"])
    assert not plan.complete(["dog", "runs"])
    assert not plan.complete(["the", "stone", "quietly"])
    assert not plan.prefix_possible(["unknown"])


def test_enumerator_state_hook_prunes_before_spending_node_budget():
    from llm_palindrome.exhaustive import enumerate_palindromes
    from llm_palindrome.search import WordTries

    stats = {}
    results = list(enumerate_palindromes(
        WordTries(["a", "aa", "aba"]), max_letters=8,
        allow_state=lambda left, right: "aa" not in left + right, stats=stats))
    assert stats["state_pruned"] > 0
    assert stats["nodes"] >= stats["yielded"]
    assert all("aa" not in words for words in results)


def test_sibling_scorer_changes_order_not_complete_result_set():
    from llm_palindrome.exhaustive import enumerate_palindromes
    from llm_palindrome.search import WordTries

    class PreferShort:
        def word_delta(self, left, right, placement, word, growth):
            return -len(word)

    tries = WordTries(["a", "aa", "aba"])
    plain = list(enumerate_palindromes(tries, max_letters=7,
                                       node_budget=100000))
    ranked = list(enumerate_palindromes(tries, max_letters=7,
                                        node_budget=100000,
                                        scorer=PreferShort()))
    assert {tuple(row) for row in plain} == {tuple(row) for row in ranked}
    assert plain != ranked

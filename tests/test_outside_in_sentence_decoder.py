from llm_palindrome.sentence_plan import SentencePlan


def test_plan_keeps_a_sentence_prefix_and_suffix_as_separate_constraints():
    plan = SentencePlan({"they": {"PRON"}, "see": {"VERB"}, "stars": {"NOUN"}},
                        {("PRON", "VERB", "NOUN")}, min_words=3, max_words=3)
    assert plan.prefix_possible(("they", "see"))
    assert plan.suffix_possible(("see", "stars"))
    assert not plan.prefix_possible(("see", "stars"))

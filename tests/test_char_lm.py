from llm_palindrome.char_lm import CharacterNgram, consume_residual, rank_constrained


def test_residual_filter_is_exact_and_keeps_intact_fragments():
    assert consume_residual("quiet room", "quietroomXYZ") == ("xyz", False)
    assert consume_residual("quiet rooms", "quietroom") == ("s", True)
    assert rank_constrained("the ", ["quiet room", "mirror words", "quiet rooms"],
                            "quietroomXYZ", CharacterNgram(["the quiet room"]))[0].text == "quiet room"


def test_character_model_is_deterministic_and_provenance_is_explicit():
    model = CharacterNgram(["the baker reads the letter", "the caller answers"])
    first = rank_constrained("the ", ["baker reads", "zzzz"], "bakerreads",
                             model)
    second = rank_constrained("the ", ["baker reads", "zzzz"], "bakerreads",
                              model)
    assert first == second
    assert first[0].provenance == "character-ngram-local-corpus"

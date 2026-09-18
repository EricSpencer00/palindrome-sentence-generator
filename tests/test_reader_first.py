import pytest

from llm_palindrome.reader_first import compose, entries, hundred_word_showcase


def test_reader_first_catalogue_reflection_is_retired():
    for call in (entries, lambda: compose(["rat"]), hundred_word_showcase):
        with pytest.raises(RuntimeError, match="retired"):
            call()

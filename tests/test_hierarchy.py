from llm_palindrome.bigram import BigramModel
from llm_palindrome.hierarchy import (is_sentence_pair, render_layout,
                                      select_sentence_pairs)
from llm_palindrome.validator import is_palindrome, normalize


def test_sentence_pair_requires_both_halves_to_pass(monkeypatch):
    import llm_palindrome.hierarchy as hierarchy

    monkeypatch.setattr(hierarchy, "_tier",
                        lambda words, *_: hierarchy.SENTENCE if words[0] != "bad" else 0)
    assert is_sentence_pair(["good"], ["also"], {}, set())
    assert not is_sentence_pair(["good"], ["bad"], {}, set())


def test_render_layout_preserves_chunks_as_sentences_and_letters():
    layout = [
        {"slot": 0, "role": "left", "text": "rats live", "source": "generated"},
        {"slot": 1, "role": "centre", "text": "on no", "source": "generated"},
        {"slot": 2, "role": "right", "text": "evil star", "source": "generated"},
    ]
    text, sentences = render_layout(layout)
    assert len(sentences) == len(layout)
    assert all(sentence["text"].endswith(".") for sentence in sentences)
    assert normalize(text) == normalize("rats live on no evil star")
    assert is_palindrome(text)


def test_selector_keeps_charming_compression_but_rejects_cycles():
    pairs = [
        (["items", "draw", "award"], ["reward", "art", "smite"], "generated"),
        (["do", "do", "work"], ["work", "do", "do"], "generated"),
        (["items", "draw", "more"], ["more", "art", "items"], "generated"),
        (["items", "draw", "wonderful"], ["wonder", "rests", "inside"], "generated"),
    ]
    used, stats = select_sentence_pairs(pairs, [], target_letters=200)
    assert [row["left"] for row in used] == [
        ["items", "draw", "award"], ["items", "draw", "more"]]
    assert stats["rejected"]["adjacent_word"] == 1
    assert stats["rejected"]["bigram"] == 1
    assert stats["max_bigram_uses"] <= 2


def test_selector_caps_repeated_content_but_keeps_connectives_available():
    pairs = [
        (["level", f"draws{x}", f"item{x}"],
         [f"return{x}", f"comes{x}", f"back{x}"], "g")
        for x in range(5)
    ]
    used, stats = select_sentence_pairs(
        pairs, [], target_letters=500, max_template_uses=20,
        max_bigram_uses=20, max_content_word_uses=3)
    assert len(used) == 3
    assert stats["max_content_word_uses"] == 3

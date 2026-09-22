from experiments.paragraph_note_dialogue_composition_20260922 import (
    _segment, _sentence_trie,
)


def test_trie_segments_multiple_complete_utterances():
    sentences = {
        "spam": [{"words": ("spam",)}],
        "no": [{"words": ("no",)}],
        "stop": [{"words": ("stop",)}],
    }
    trie = _sentence_trie(sentences)
    assert ("spam", "no", "stop") in _segment(
        "spamnostop", trie, maximum_sentences=3
    )


def test_trie_rejects_incomplete_suffix():
    trie = _sentence_trie({"stop": [{"words": ("stop",)}]})
    assert _segment("stopx", trie, maximum_sentences=2) == []

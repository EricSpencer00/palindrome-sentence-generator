"""Turn a flat word sequence into sentence-cased text.

Punctuation and casing are invisible to the palindrome property (normalize()
strips them), so sentence breaks are free. The invariant that must hold:
normalize(textify(words)) == normalize(" ".join(words)).
"""
from __future__ import annotations

from typing import Sequence

from .validator import normalize


def textify(words: Sequence[str], words_per_sentence: int = 7) -> str:
    if not words:
        return ""
    sentences = []
    for i in range(0, len(words), words_per_sentence):
        chunk = list(words[i:i + words_per_sentence])
        chunk[0] = chunk[0].capitalize()
        sentences.append(" ".join(chunk) + ".")
    text = " ".join(sentences)
    assert normalize(text) == normalize(" ".join(words))
    return text

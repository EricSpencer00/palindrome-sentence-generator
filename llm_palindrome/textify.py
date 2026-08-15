"""Turn a flat word sequence into sentence-cased text.

Punctuation and casing are invisible to the palindrome property (normalize()
strips them), so sentence breaks are free. The invariant that must hold:
normalize(textify(words)) == normalize(" ".join(words)).

That freedom is worth spending deliberately. Cutting every N words puts the
period wherever the count lands — through the middle of a run that read as a
clause as often as at a place the text had already broken. A judge shown
fixed-stride sentences rejected 15 of 15 while passing 3 of 3 real ones.
`segment_at_weak_joins` spends the freedom instead on the joins the bigram
model likes least, which is where the text has already fallen apart.
"""
from __future__ import annotations

from typing import Optional, Sequence

from .validator import normalize


def segment_at_weak_joins(words: Sequence[str], bigrams,
                          sentences: int = 8) -> list[list[str]]:
    """Split `words` into `sentences` runs, breaking at the worst joins.

    Asking for more sentences than there are words gives one word each; the
    breaks are the `sentences - 1` lowest-scoring adjacent pairs, so no run is
    ever empty.
    """
    words = list(words)
    if not words:
        return []
    if sentences <= 1 or len(words) == 1:
        return [words]

    joins = [(bigrams.forward(a, b), i + 1)
             for i, (a, b) in enumerate(zip(words, words[1:]))]
    joins.sort(key=lambda item: (item[0], item[1]))
    cuts = sorted({i for _, i in joins[:sentences - 1]})

    out, start = [], 0
    for cut in cuts:
        out.append(words[start:cut])
        start = cut
    out.append(words[start:])
    return [run for run in out if run]


def segment_at_units(units: Sequence[str], min_unit_words: int = 3) -> list[list[str]]:
    """Give every attested unit of `min_unit_words` or more its own sentence.

    Once clause-length n-grams are in the trie, the output stops being uniform:
    it holds runs of real English — mined whole from a corpus — separated by
    the short filler the letter constraint forces. Those runs are the only
    spans in the text with a guarantee attached, and the search already knows
    where they are, because it chose them. Cutting anywhere else slices through
    the one part worth showing.
    """
    out: list[list[str]] = []
    buffer: list[str] = []
    for unit in units:
        words = unit.split()
        if len(words) >= min_unit_words:
            if buffer:
                out.append(buffer)
                buffer = []
            out.append(words)
        else:
            buffer.extend(words)
    if buffer:
        out.append(buffer)
    return [run for run in out if run]


def textify(words: Sequence[str], words_per_sentence: int = 7,
            segments: Optional[Sequence[Sequence[str]]] = None) -> str:
    """Render words as sentence-cased text.

    `segments` overrides the fixed stride with runs chosen by the caller —
    normally `segment_at_weak_joins`.
    """
    if not words:
        return ""
    if segments is None:
        segments = [list(words[i:i + words_per_sentence])
                    for i in range(0, len(words), words_per_sentence)]

    rendered = []
    for chunk in segments:
        chunk = list(chunk)
        if not chunk:
            continue
        chunk[0] = chunk[0].capitalize()
        rendered.append(" ".join(chunk) + ".")
    text = " ".join(rendered)
    assert normalize(text) == normalize(" ".join(words))
    return text

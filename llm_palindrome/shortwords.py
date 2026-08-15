"""The one- and two-letter words the search is allowed to use.

Short words fit any overhang, so the search reaches for them whenever the
letters get awkward — they are the cheapest way to pay a debt. That makes this
the one corner of the vocabulary where a frequency cut fails badly. wordfreq's
top 100k contains "bn", "cu", "eb", "ek", "fo", "iw" and "ht" because web text
contains them, and the generator used every one: 52.6% of its words were one or
two letters against real English's 18.5%, and the excess was not English.

A frequency threshold cannot fix it. Brown at 20+ occurrences still admits
"aj", "du" and every bare initial, because an edited corpus is full of names
and abbreviations. So the list is explicit and short enough to read, which also
makes it arguable — if a word here is wrong, it is wrong visibly.

`REAL_SINGLE_LETTERS` in generate.py made this same judgement for one letter
already. This extends it to two.
"""
from __future__ import annotations

# Ordinary English words. Interjections are included where they are written
# this way in edited prose ("oh", "ah"); dialect spellings and abbreviations
# are not, however common the web finds them.
REAL_SHORT_WORDS = frozenset({
    # one letter
    "a", "i",
    # pronouns and determiners
    "he", "me", "my", "we", "us", "it", "an",
    # verbs and auxiliaries
    "am", "be", "do", "go", "is",
    # prepositions and conjunctions
    "as", "at", "by", "if", "in", "of", "on", "or", "to", "up",
    # adverbs and particles
    "no", "so",
    # interjections that appear in edited prose
    "ah", "oh",
    # everyday nouns and abbreviations that are genuinely written this way
    "ma", "pa", "tv", "ok",
})


def is_real_short(word: str) -> bool:
    """True unless `word` is a one- or two-letter string that is not a word."""
    return len(word) > 2 or word in REAL_SHORT_WORDS

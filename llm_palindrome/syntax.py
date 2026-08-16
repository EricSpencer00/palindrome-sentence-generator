"""Does a half have the SHAPE of an English sentence?

Attested joins made the walk's output stop looking like word salad and did not
make it grammatical: "to host a test on" reads and "not set at so hot" does
not, and every join in both is one English attests. The difference is syntax,
and a bigram over words cannot see it.

A tagged corpus can. Brown gives every word the tags it is attested with, and
every sentence a tag sequence. Two questions can then be asked of a half, and
they are different questions:

  shaped     is there a tag reading of these words that a whole sentence in
             the corpus also has? — sentence-shaped, the strict test
  plausible  is every tag trigram in some reading one the corpus contains? —
             locally well-formed, the loose one

Measured on 179 walked pairs, and the numbers say which one is worth having.
`plausible` passes 165 of 179 — it is not a filter, it is a formality.
`shaped` passes more than half and still admits "draw at left one man".
`sentence_like`, which adds a verb and a subject-shaped opening, passes 3, and
on the hand cases it agrees with reading them: "to host a test on" and "no rats
live" pass, "not set at so hot" and "draw at left one man" fail.

It is wrong in the direction you would expect a tag test to be wrong. "Lived on
decaf" and "ima lasagna hog" both fail it, and both are readable English of the
elliptical kind a corpus of edited prose does not contain. So it narrows what a
person reads and it must not decide what ships — the project's rule about
proxies, which four of them have now broken.

The tag table is a dictionary of what each word CAN be, so a word Brown never
saw has no tags and every test fails closed. That is the right default here:
the vocabulary the walk draws from is frequency-ranked English, and a word
missing from a million tagged words is unusual enough to be worth losing.
"""
from __future__ import annotations

from collections import Counter
from itertools import product
from typing import Iterable, Optional, Sequence

# More than this many tag combinations and the half is not worth resolving:
# the product grows with ambiguity, and an eight-word half of three-tag words
# is 6,561 readings to test.
MAX_READINGS = 20000


def tag_table(tagged_sentences: Iterable[Sequence[tuple[str, str]]]
              ) -> dict[str, frozenset[str]]:
    """word -> the tags the corpus attests for it, lowercased."""
    seen: dict[str, set[str]] = {}
    for sentence in tagged_sentences:
        for word, tag in sentence:
            seen.setdefault(word.lower(), set()).add(tag)
    return {w: frozenset(tags) for w, tags in seen.items()}


def sentence_shapes(tagged_sentences: Iterable[Sequence[tuple[str, str]]],
                    min_words: int = 3, max_words: int = 9,
                    drop_tags: frozenset = frozenset({".", "X"})
                    ) -> set[tuple[str, ...]]:
    """Tag sequences of whole corpus sentences in the length band."""
    out: set[tuple[str, ...]] = set()
    for sentence in tagged_sentences:
        shape = tuple(t for _, t in sentence if t not in drop_tags)
        if min_words <= len(shape) <= max_words:
            out.add(shape)
    return out


def tag_trigrams(tagged_sentences: Iterable[Sequence[tuple[str, str]]],
                 drop_tags: frozenset = frozenset({".", "X"})) -> Counter:
    """Every three-tag run the corpus contains, counted.

    Sentence boundaries are not padded. A half is a fragment of prose as often
    as it is a sentence, and asking whether its interior is well-formed is a
    different question from asking whether it could stand alone.
    """
    out: Counter = Counter()
    for sentence in tagged_sentences:
        shape = [t for _, t in sentence if t not in drop_tags]
        for i in range(len(shape) - 2):
            out[tuple(shape[i:i + 3])] += 1
    return out


def readings(words: Sequence[str], table: dict[str, frozenset[str]],
             limit: int = MAX_READINGS) -> list[tuple[str, ...]]:
    """Every tag sequence these words could carry, or [] if any is unknown."""
    pools = []
    total = 1
    for word in words:
        tags = table.get(word)
        if not tags:
            return []
        total *= len(tags)
        if total > limit:
            return []
        pools.append(sorted(tags))
    return [tuple(r) for r in product(*pools)]


def shaped(words: Sequence[str], table: dict[str, frozenset[str]],
           shapes: set[tuple[str, ...]]) -> bool:
    """Could these words be read as a whole sentence the corpus attests?"""
    return any(r in shapes for r in readings(words, table))


def plausible(words: Sequence[str], table: dict[str, frozenset[str]],
              trigrams, min_count: int = 1) -> bool:
    """Is there a reading whose every tag trigram the corpus contains?

    Two-word halves have no trigram and pass vacuously; they are excluded
    elsewhere, by the word floor, because a two-word half is a fragment
    whatever its tags say.
    """
    for reading in readings(words, table):
        if all(trigrams[reading[i:i + 3]] >= min_count
               for i in range(len(reading) - 2)):
            return True
    return False


# Tags a clause can open on and still have a subject. Shared with
# `compose.SUBJECT_TAGS` in intent; ADV is added because "still it held" opens
# on one and is a sentence.
OPENING_TAGS = frozenset({"PRON", "DET", "NOUN", "ADJ", "NUM", "ADV"})


def sentence_like(words: Sequence[str], table: dict[str, frozenset[str]],
                  shapes: set[tuple[str, ...]]) -> bool:
    """Could these words be a sentence with a subject and a verb?

    The three conditions have to hold in the SAME reading. Checking them
    separately passes "draw at left one man" — there is a reading with a verb
    and a reading with an attested shape, and they are not the same reading.
    """
    for reading in readings(words, table):
        if (reading in shapes and "VERB" in reading
                and reading[0] in OPENING_TAGS):
            return True
    return False


_BROWN: Optional[tuple] = None


def brown_tables(min_words: int = 3, max_words: int = 9) -> tuple:
    """(table, shapes, trigrams) from Brown, built once per process."""
    global _BROWN
    if _BROWN is None:
        from nltk.corpus import brown
        tagged = list(brown.tagged_sents(tagset="universal"))
        _BROWN = (tag_table(tagged),
                  sentence_shapes(tagged, min_words, max_words),
                  tag_trigrams(tagged))
    return _BROWN

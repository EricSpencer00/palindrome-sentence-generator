"""Keep the public generator from emitting slurs and porn vocabulary.

wordfreq ranks by web frequency, so those words sit comfortably inside the top
30k. The search optimises for letter fit and bigram plausibility and has no
notion of taste, so the only reliable place to intervene is the vocabulary:
a word that never enters the trie can never appear in a palindrome.

Matching is whole-word. Substring matching would remove "class", "assist" and
"analysis", which is both wrong and conspicuous.

Whole-word matching means every inflection has to be listed, and hand-listing
them does not work: the blocklist carried "rape" and "raped" but not "rapes",
and "rapes" duly appeared 24 times in a 2000-palindrome sample. So the forms
are generated from each base word instead.

Generating them creates the opposite hazard, because a slur can be a prefix of
an ordinary word: "spic" yields "spicy" and "spiced". ALLOWED_ANYWAY is the
answer to both — it is applied after expansion, and it is populated by actually
diffing the filter against the top-30k list rather than by guessing.

**Only EXTRA_BLOCKED is expanded.** better_profanity's entries stay exact
matches. Expanding them too was tried and measured: it removed 125 further
words from the top-30k list, among them "her", "killed", "tested", "sober",
"weeds" and "assessing". That list contains short and mangled stems (its
entries are VaryingString objects, some spelled "s.h.i.t."), and inflecting a
noisy list multiplies the noise. The curated list is small enough to audit,
which is what makes expanding it safe.
"""
from __future__ import annotations

import warnings
from functools import lru_cache
from typing import Iterable

# Slurs the general-purpose blocklist misses, plus terms that are innocuous in
# isolation but only reach the top-30k through adult sites.
EXTRA_BLOCKED = {
    "xxx", "porn", "porno", "pornhub", "xnxx", "xvideos", "sex", "sexo",
    "sexy", "milf", "hentai", "escort", "escorts", "nude", "nudes", "naked",
    "boobs", "tits", "penis", "vagina", "orgasm", "erotic", "fetish", "bdsm",
    "incest", "rape", "raped", "rapist", "molest", "pedo", "lolita",
    "nigger", "nigga", "faggot", "fag", "kike", "spic", "chink", "wetback",
    "tranny", "retard", "retarded", "nazi", "hitler", "suicide", "kill",
}

# Words the blocklist flags that are ordinary English in every other context.
# The second group are collisions produced by inflecting a blocked base: each
# one was found by diffing the expanded filter against the top-30k vocabulary,
# not predicted. Re-run tests/test_safe_vocab.py::test_no_ordinary_word_lost
# after touching EXTRA_BLOCKED — a new base can introduce a new collision.
ALLOWED_ANYWAY = {
    "hell", "damn", "crap", "sucks", "lust", "gay", "lesbian",
    # Collisions from inflecting "spic". Nothing else in EXTRA_BLOCKED
    # generates an ordinary word; this was checked against the whole list.
    "spicy", "spiced", "spicing",
}


def _inflections(word: str) -> set[str]:
    """Regular English forms of a base word: plural, past, participle, agent.

    Deliberately regular-only. Irregular forms are rare among the bases here,
    and a rule general enough to catch them would sweep in far more ordinary
    vocabulary than it removed.
    """
    forms = {word}
    if word.endswith("y") and len(word) > 2 and word[-2] not in "aeiou":
        stem = word[:-1]
        forms.update({stem + "ies", stem + "ied", stem + "ier", stem + "iest"})
    elif word.endswith(("s", "x", "z", "ch", "sh")):
        forms.add(word + "es")
    else:
        forms.add(word + "s")

    if word.endswith("e"):
        stem = word[:-1]
        forms.update({stem + "ed", stem + "es", stem + "ing", stem + "er",
                      stem + "ers", stem + "y"})
    else:
        forms.update({word + "ed", word + "ing", word + "er", word + "ers",
                      word + "y"})
        # consonant-vowel-consonant doubles the final letter: fag -> fagging
        if (len(word) >= 3 and word[-1] not in "aeiouwxy"
                and word[-2] in "aeiou" and word[-3] not in "aeiou"):
            doubled = word + word[-1]
            forms.update({doubled + "ed", doubled + "ing", doubled + "er",
                          doubled + "ers", doubled + "y"})
    return forms


def _expand(words: Iterable[str]) -> set[str]:
    out: set[str] = set()
    for w in words:
        out |= _inflections(w)
    return out


@lru_cache(maxsize=1)
def _blocklist() -> frozenset[str]:
    # Curated bases are inflected; third-party entries are matched as given.
    words = _expand(EXTRA_BLOCKED)
    try:
        from better_profanity import profanity
        profanity.load_censor_words()
        # Entries are VaryingString objects, not str, and some are spelled with
        # separators ("s.h.i.t."). Take the text and also a stripped form.
        for entry in profanity.CENSOR_WORDSET:
            text = str(entry).lower()
            words.add(text)
            words.add("".join(ch for ch in text if ch.isalpha()))
    except Exception:
        # The curated set above still applies, but it is 40 words against the
        # package's 900. This vocabulary reaches a public endpoint, so a
        # machine that quietly falls back to the short list is worth hearing
        # about — the failure is otherwise invisible until something ships.
        warnings.warn(
            "better_profanity is unavailable; vocabulary filtering falls back "
            f"to {len(EXTRA_BLOCKED)} curated words. Install it before serving "
            "generated text publicly.", RuntimeWarning, stacklevel=2)
    # Subtract the allowlist after expansion: an inflection of a blocked base
    # is exactly what ALLOWED_ANYWAY has to be able to rescue.
    return frozenset(words - _expand(ALLOWED_ANYWAY))


def is_allowed(word: str) -> bool:
    return word.lower() not in _blocklist()


def safe_vocab(words: Iterable[str]) -> list[str]:
    return [w for w in words if is_allowed(w)]

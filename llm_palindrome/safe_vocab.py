"""Keep the public generator from emitting slurs and porn vocabulary.

wordfreq ranks by web frequency, so those words sit comfortably inside the top
30k. The search optimises for letter fit and bigram plausibility and has no
notion of taste, so the only reliable place to intervene is the vocabulary:
a word that never enters the trie can never appear in a palindrome.

Matching is whole-word. Substring matching would remove "class", "assist" and
"analysis", which is both wrong and conspicuous.
"""
from __future__ import annotations

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
ALLOWED_ANYWAY = {"hell", "damn", "crap", "sucks", "lust", "gay", "lesbian"}


@lru_cache(maxsize=1)
def _blocklist() -> frozenset[str]:
    words = set(EXTRA_BLOCKED)
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
        # The curated set above still applies if the package is unavailable.
        pass
    return frozenset(words - ALLOWED_ANYWAY)


def is_allowed(word: str) -> bool:
    return word.lower() not in _blocklist()


def safe_vocab(words: Iterable[str]) -> list[str]:
    return [w for w in words if is_allowed(w)]

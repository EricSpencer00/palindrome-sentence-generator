"""A phrase inventory: attested n-grams the search consumes atomically.

This is the last idea `docs/training.md` left standing. Reranking the search's
output is exhausted (`oracle_bound.py` — the curve flattens, which bounds every
possible judge), widening the search makes the text worse (`diversity_sweep.py`
— readability falls monotonically), and length buys nothing (`length_sweep.py`
— the output sits on the word-salad line from 71 letters to 1197). All three
share a cause: the trie holds single words, so every join in the output is a
join the scorer had to discover, and a scorer with a two-word horizon cannot
discover more than two words of coherence at a time.

An inventory changes what a join IS. "new york" enters the trie as one unit
keyed on `newyork`, so when the search places it, the join inside it is not
something the scorer got right — it is something the corpus already attested,
and the search could not have broken it without choosing a different unit.

The vocabulary filter applies to both words. `safe_vocab` exists because the
frequency-ranked list carries slurs, and an inventory drawn from the same
corpus is a second door into the same vocabulary.
"""
from __future__ import annotations

from typing import Iterable, Optional


def parse_bigram_file(path: str) -> dict[tuple[str, str], int]:
    """Read Norvig's count_2w.txt: 'word1 word2<TAB>count' per line.

    Lines that are not exactly two whitespace-separated words are skipped
    rather than repaired — the file is a corpus artifact and a line that does
    not parse is a line whose two words are not knowable.
    """
    pairs: dict[tuple[str, str], int] = {}
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 2:
                continue
            words = parts[0].split()
            if len(words) != 2:
                continue
            try:
                count = int(parts[1])
            except ValueError:
                continue
            pairs[(words[0].lower(), words[1].lower())] = count
    return pairs


def build_inventory(path: str, vocab: Iterable[str], top_n: int = 20000,
                    min_count: int = 1,
                    pairs: Optional[dict[tuple[str, str], int]] = None) -> list[str]:
    """The `top_n` most frequent attested pairs, both words inside `vocab`.

    Returned as spaced phrases ("new york"); `search.unit_letters` is what
    turns one into the run of letters it fills.
    """
    allowed = set(vocab)
    if pairs is None:
        pairs = parse_bigram_file(path)

    kept = [((a, b), n) for (a, b), n in pairs.items()
            if n >= min_count and a in allowed and b in allowed
            and a.isalpha() and b.isalpha()]
    kept.sort(key=lambda item: (-item[1], item[0]))
    return [f"{a} {b}" for (a, b), _ in kept[:top_n]]


def mine_ngrams(corpus: Iterable[str], n: int, min_count: int = 2,
                vocab: Optional[Iterable[str]] = None,
                top_n: Optional[int] = None) -> list[str]:
    """Word n-grams occurring at least `min_count` times in running text.

    Bigram units did not buy sentences — two generations, zero of 25 fragments
    judged coherent — and the reason is arithmetic. A two-word unit guarantees
    one attested join; a sentence needs five or six consecutive ones, and the
    search has to find the rest itself against a letter constraint.

    A 6-gram lifted from a corpus is a different kind of object. It is not an
    inference from pair counts that the words go together — it is a fragment of
    English somebody actually wrote, and placing it whole puts that fragment in
    the output intact.

    Tokens must be alphabetic: numbers and punctuation have no letters the
    mirror can use, and a unit carrying them could never be placed.
    """
    from collections import Counter

    allowed = set(vocab) if vocab is not None else None
    counts: Counter[str] = Counter()
    for line in corpus:
        words = [w.lower() for w in line.split()]
        for i in range(len(words) - n + 1):
            gram = words[i:i + n]
            if not all(w.isalpha() for w in gram):
                continue
            if allowed is not None and not all(w in allowed for w in gram):
                continue
            counts[" ".join(gram)] += 1

    kept = [(g, c) for g, c in counts.items() if c >= min_count]
    kept.sort(key=lambda item: (-item[1], item[0]))
    if top_n is not None:
        kept = kept[:top_n]
    return [g for g, _ in kept]


def mine_sentences(corpus: Iterable[str], min_words: int = 3, max_words: int = 8,
                   vocab: Optional[Iterable[str]] = None,
                   top_n: Optional[int] = None) -> list[str]:
    """Whole sentences, start to end, short enough for the mirror to repay.

    `mine_ngrams` produces spans from the middle of sentences, and generation 5
    showed what that costs: isolating one as a sentence yields "Was unable to
    make." — grammatical, and trailing off. A judge rejected 16 of 16.

    A unit that runs from a sentence's start to its end has a beginning and an
    end because it inherited them. And unlike an n-gram it needs no repetition
    to justify it: an n-gram is trusted because it recurs, but a sentence
    occurring once is already a sentence somebody wrote.

    The word cap is the palindrome's constraint, not English's. Every letter
    placed has to be mirrored by letters that are also English, so a unit's
    cost grows with its length and long sentences simply never close.
    """
    import re

    allowed = set(vocab) if vocab is not None else None
    out: list[str] = []
    seen: set[str] = set()
    for line in corpus:
        for raw in re.split(r"(?<=[.!?])\s+", line):
            words = [w.lower() for w in raw.strip().rstrip(".!?").split()]
            if not (min_words <= len(words) <= max_words):
                continue
            if not all(w.isalpha() for w in words):
                continue
            if allowed is not None and not all(w in allowed for w in words):
                continue
            sentence = " ".join(words)
            if sentence not in seen:
                seen.add(sentence)
                out.append(sentence)
            if top_n is not None and len(out) >= top_n:
                return out
    return out


def build_units(vocab: Iterable[str], inventory: Iterable[str]) -> list[str]:
    """Single words and phrases together, which is the point.

    The search keeps both: a phrase is only usable when the overhang happens to
    admit its whole run of letters, and a search with no single words to fall
    back on would close far less often.
    """
    return sorted(set(vocab) | set(inventory))

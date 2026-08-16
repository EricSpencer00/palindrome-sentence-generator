"""Put the spaces back into a letters-only palindrome.

The canon is stored normalised, because the novelty check compares palindromes
by their letters and spacing is noise there. Everywhere else it is fatal:
`harvest` needs a word boundary at the letter midpoint to make a mirror-pair,
so a text with no boundaries at all can only be a centre — and a paragraph
takes one centre. The result was that the only readable letter-level material
this project has (120 verified palindromes) contributed nothing to assembly,
while 20,000 unreadable pairs from the hunts contributed everything.

Recovery is word segmentation under a unigram model: choose the reading that
maximises the summed log-probability of its words, with a per-word cost so "a"
and "i" cannot tile an arbitrary run for free. Dynamic programming over letter
positions, so an ambiguous 40-letter run costs O(n^2) rather than enumerating
the exponentially many readings.

Ambiguity is genuine and not always resolvable — "madaminedenimadam" is both
"madam in eden im adam" and "mad amine den im adam", and only the first is the
palindrome anyone means. Recovery is measured against recorded spellings in
tests/test_respace.py rather than trusted.
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Iterable, Sequence

# Each additional word must pay for itself. Without this the model prefers
# "a man a plan" spelled as eleven one-letter words, since every split adds
# probability mass rather than spending it.
WORD_COST = 9.0

# Longest word considered at any position; the vocabulary's longest entry is
# well under this and the cap keeps the inner loop short.
MAX_WORD = 20

# Short forms the canon spells that `shortwords.REAL_SHORT_WORDS` omits. Kept
# as an explicit list so the harsh two-letter rule stays harsh everywhere else.
CANON_SHORT = frozenset({"im"})


def canon_vocab(n: int = 60000) -> list[str]:
    """The vocabulary for RECOVERING spellings — not for generating text.

    `generate.build_vocab` deliberately withholds words: `safe_vocab` drops
    what must never reach a generated public output, and `is_real_short` drops
    two-letter strings the frequency list contains but no reader accepts. Both
    are right for a search that invents sentences.

    `safe_vocab` does not apply here. Recovery only decides where the spaces
    fall in a palindrome that is already stored, already verified, and already
    published as letters; withholding "god" does not remove it from
    "dogeeseseegod", it only makes the line unreadable.

    `is_real_short` still does. It is not a policy about output, it is the
    claim that "jv" and "kw" are not words — and the frequency list offers both
    as one-letter-at-a-time filler for any run that will not segment. The canon
    needs one short form the allowlist omits, "im", as in "madam in eden im
    adam"; it is admitted by name rather than by loosening the rule.

    The rank is raised to 60k because the canon leans on names and archaisms —
    elba, naomi, ere — that sit past the 30k the generator uses.
    """
    from wordfreq import top_n_list

    from llm_palindrome.shortwords import is_real_short
    return [w for w in top_n_list("en", n)
            if w.isalpha() and w.isascii()
            and (is_real_short(w) or w in CANON_SHORT)]


@lru_cache(maxsize=None)
def _zipf(word: str) -> float:
    from wordfreq import zipf_frequency
    return zipf_frequency(word, "en")


def unigram_score(words: Sequence[str]) -> float:
    """Summed word log-probability, less a fixed cost per word.

    Zipf frequency is already a log scale, so it stands in for log P directly.
    A word the frequency list has never seen scores far below anything real,
    which keeps invented segments out without a separate check.
    """
    total = 0.0
    for w in words:
        z = _zipf(w)
        total += (z if z > 0 else -10.0) - WORD_COST
    return total


def respace_k(letters: str, vocab: Iterable[str], k: int = 8) -> list[list[str]]:
    """The k most probable readings of a letter run, best first.

    `respace` returns only the winner, which is right for recovering a canon
    entry — there the true spelling exists and the model is guessing it. Mining
    asks a different question: whether ANY reading is good English. "drawa"
    reversed reads best as "award", while the reading English attests is "a
    ward", and only k-best offers it.

    How much this is worth was measured rather than assumed: over the attested
    phrase list, k=8 recovers 4 more attested-both-halves pairs out of 198, a
    2% gain. Small — the single-best reading is usually the only one.

    Standard k-best dynamic programming: each position keeps its k best
    partial readings rather than one, so the cost is k times the single-best
    pass and never the exponential enumeration of all segmentations.
    """
    letters = "".join(c.lower() for c in letters if c.isalpha())
    if not letters or k < 1:
        return []
    words = vocab if isinstance(vocab, (set, frozenset)) else {
        w.lower() for w in vocab if w and w.isalpha()}

    n = len(letters)
    # best[i] holds up to k (score, reading) for letters[:i], best first.
    best: list[list[tuple[float, list[str]]]] = [[] for _ in range(n + 1)]
    best[0] = [(0.0, [])]
    for i in range(1, n + 1):
        found: list[tuple[float, list[str]]] = []
        for j in range(max(0, i - MAX_WORD), i):
            if not best[j]:
                continue
            piece = letters[j:i]
            if piece not in words:
                continue
            cost = unigram_score([piece])
            for score, reading in best[j]:
                found.append((score + cost, reading + [piece]))
        found.sort(key=lambda sr: -sr[0])
        best[i] = found[:k]
    return [reading for _, reading in best[n]]


# Reward per attested join when choosing among k-best readings. Swept, not
# picked: 0 fixes nothing, 1-4 fixes three broken canon entries with no
# regressions, and 6+ starts breaking good ones ("borrow or rob" becomes "bor
# row or rob") because more words mean more joins to count.
JOIN_BONUS = 2.0


def respace_attested(letters: str, vocab: Iterable[str],
                     attested: "set[tuple[str, str]]",
                     k: int = 24) -> list[str]:
    """Respace, breaking ties toward readings English has been seen to make.

    A unigram model scores words in isolation, so it cannot tell "for ajar"
    from "for a jar" — and blind judging found that 14 of 33 rejected centres
    were correct palindromes wrecked by exactly that. Attestation is the
    evidence a unigram model lacks.

    Worth three of ten measured. The rest fail for a different reason: their
    correct readings are absent from the k-best at any k up to 160, so beam
    width is not what is holding them.
    """
    readings = respace_k(letters, vocab, k=k)
    if not readings:
        return []
    return max(readings, key=lambda r: unigram_score(r) + JOIN_BONUS * sum(
        (a, b) in attested for a, b in zip(r, r[1:])))


def respace(letters: str, vocab: Iterable[str]) -> list[str]:
    """The most probable reading of a run of letters, as a word list.

    Returns [] when no reading exists at all — the caller decides whether that
    is a corrupt entry or a palindrome built from words outside the vocabulary.
    """
    letters = "".join(c.lower() for c in letters if c.isalpha())
    if not letters:
        return []
    # Mining pairs calls this once per attested phrase, so normalising a 60k
    # vocabulary on every call is the whole cost of the run. A caller that
    # already holds a normalised set hands it straight through.
    if isinstance(vocab, (set, frozenset)):
        words = vocab
    else:
        words = {w.lower() for w in vocab if w and w.isalpha()}

    n = len(letters)
    best = [-math.inf] * (n + 1)
    back = [-1] * (n + 1)
    best[0] = 0.0
    for i in range(1, n + 1):
        for j in range(max(0, i - MAX_WORD), i):
            if best[j] == -math.inf:
                continue
            piece = letters[j:i]
            if piece not in words:
                continue
            cand = best[j] + unigram_score([piece])
            if cand > best[i]:
                best[i] = cand
                back[i] = j
    if best[n] == -math.inf:
        return []

    out: list[str] = []
    i = n
    while i > 0:
        j = back[i]
        out.append(letters[j:i])
        i = j
    out.reverse()
    return out

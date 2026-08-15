"""Mirror-pairs mined from attested English, rather than proposed by a search.

Every earlier attempt to get readable units generated candidates and then tried
to rank them: the exhaustive hunt walks a vocabulary and scores closures with
GPT-2, and the README records that the LM score, bigram-join attestation,
vocabulary filters and edge-joins each failed against judge verdicts. The
walk's own top-25 is "no it cab action" twenty-five times over.

Mining inverts that. Take the left half from a bigram English has already
attested — it is readable because it occurred — and ask only whether the
mirrored letter run also reads, which `respace` decides by segmentation under a
unigram model. Nothing is proposed and nothing needs ranking for readability:
both halves are English or the pair does not exist. Over 272k attested bigrams
this yields tens of thousands of pairs in about two seconds.

The mirror cost has not been evaded. It shows up as the yield: most attested
phrases have no readable mirror, and the ones that do are short. What changes
is that the cost is paid by discarding candidates English vouched for, instead
of by ranking candidates nothing vouched for.

Mining is GENERATION — it produces phrases this project has never held — so it
takes `generate.build_vocab`, not `respace.canon_vocab`. The latter omits
`safe_vocab` because recovering the spelling of a stored palindrome is not
generation; a trial run that used it here put "not raped" in the output.
"""
from __future__ import annotations

from typing import Iterable, Iterator, Sequence

from llm_palindrome.respace import respace, respace_k
from llm_palindrome.validator import is_palindrome


def attested_phrases(path: str, vocab: Iterable[str]) -> Iterator[str]:
    """Lowercased two-word phrases from a 'word1 word2<TAB>count' file.

    Ordered by attestation, most-attested first, so a caller that truncates
    keeps the phrases English uses most.
    """
    allowed = {w.lower() for w in vocab if w and w.isalpha()}
    rows: list[tuple[int, str]] = []
    for line in open(path):
        try:
            phrase, count = line.rstrip("\n").split("\t")
        except ValueError:
            continue
        words = phrase.lower().split()
        if len(words) != 2 or not all(w.isalpha() and w in allowed
                                      for w in words):
            continue
        try:
            rows.append((int(count), " ".join(words)))
        except ValueError:
            continue
    rows.sort(key=lambda r: -r[0])
    for _, phrase in rows:
        yield phrase


def attested_ngrams(path: str, vocab: Iterable[str], n: int) -> Iterator[str]:
    """Lowercased n-word phrases from the WikiText n-gram file.

    A mined half can be no longer than the phrase it came from, so bigrams
    alone produce an inventory of two-word fragments. Trigrams raise the
    ceiling; 4-grams do not, because at that length nothing mirrors — measured
    yields of both-attested pairs are 131 from bigrams, 27 from trigrams and 0
    from 4-grams, which is the 3.3 bits per letter arriving as a length curve.
    """
    import json

    with open(path) as handle:
        data = json.load(handle)
    allowed = {w.lower() for w in vocab if w and w.isalpha()}
    for phrase in data.get(str(n), []):
        words = phrase.lower().split()
        if len(words) == n and all(w.isalpha() and w in allowed
                                   for w in words):
            yield " ".join(words)


def attested_bigrams(path: str) -> set[tuple[str, str]]:
    """Every adjacent word pair the count file records, lowercased."""
    out: set[tuple[str, str]] = set()
    for line in open(path):
        phrase = line.rstrip("\n").split("\t")[0]
        words = phrase.lower().split()
        if len(words) == 2:
            out.add((words[0], words[1]))
    return out


def reads_as_attested(words: Sequence[str],
                      attested: set[tuple[str, str]]) -> bool:
    """True when every join in `words` is one English has been seen to make.

    The left half of a mined pair passes this by construction. The right half
    rarely does — 157 of 3,922 — and that ratio is the mirror cost stated as a
    yield: a phrase whose letters reverse into another attested phrase is a
    coincidence English affords about three times in a hundred.
    """
    return all((a, b) in attested for a, b in zip(words, words[1:]))


def mine_pairs(phrases: Iterable[str], vocab: Sequence[str],
               min_letters: int = 6, max_letters: int = 16,
               min_words: int = 1, min_word_letters: int = 1,
               side: str = "left",
               prefer_attested: "set[tuple[str, str]] | None" = None,
               k: int = 8
               ) -> Iterator[tuple[list[str], list[str]]]:
    """Attested phrases whose mirror also reads, as (left, right) word lists.

    `right` is in reading order, which is where `paragraphs.assemble` places it
    in the closing half.

    `side` says which half the attested phrase becomes. It matters because
    `respace` returns one reading: when a mirror has several and the unigram
    model prefers an unattested one, the attested pair is lost — and lost
    quietly, because a pair was still produced. Running both directions and
    taking the union asks the question from both ends.
    """
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")

    allowed = frozenset(w.lower() for w in vocab if w and w.isalpha())
    seen: set[tuple[str, str]] = set()

    for phrase in phrases:
        given = phrase.lower().split()
        if not given or not all(w in allowed for w in given):
            continue
        letters = "".join(given)
        if not min_letters <= len(letters) <= max_letters:
            continue

        if prefer_attested is None:
            mirrored = respace(letters[::-1], allowed)
        else:
            # The model's favourite reading is usually the only one, but where
            # it is not, an attested reading beats a more probable unattested
            # one — that is the whole quality signal. Worth 4 pairs in 198.
            readings = respace_k(letters[::-1], allowed, k=k)
            # A one-word reading has no joins, so `reads_as_attested` is
            # vacuously true for it and would always win the preference:
            # "award" beat "a ward", which is the reading English attests and
            # the only one that survives a min_words of 2. Attestation is only
            # evidence where there is a join to attest.
            mirrored = next(
                (r for r in readings
                 if len(r) > 1 and reads_as_attested(r, prefer_attested)),
                readings[0] if readings else [])
        if not mirrored:
            continue
        left, right = ((given, mirrored) if side == "left"
                       else (mirrored, given))
        # A pair whose halves are the same words reversed ("level a" / "a
        # level") reads as a stutter rather than a turn.
        if right == left or sorted(right) == sorted(left):
            continue
        if len(left) < min_words or len(right) < min_words:
            continue
        if min(len(w) for w in left + right) < min_word_letters:
            continue

        key = (" ".join(left), " ".join(right))
        if key in seen:
            continue
        seen.add(key)
        # Cheap to assert and the whole point of the module; a segmentation
        # that dropped or added a letter would silently break assembly.
        if not is_palindrome(key[0] + " " + key[1]):
            continue
        yield left, right

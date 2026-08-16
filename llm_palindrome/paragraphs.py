"""Paragraph-length palindromes, assembled from short units.

The mirror costs 3.296 bits per free letter (measured, stable across span
lengths), which is why every attempt at paragraph-length palindromic PROSE in
this repository produced English glued to gibberish: past ~30 letters the
coherent feasible set of free-running text is effectively empty, and no scorer
can select what does not exist.

But a paragraph does not have to be free-running. Two kinds of unit pay the
mirror constraint entirely within themselves:

  center   a sentence that is itself a palindrome
  pair     two sentences whose letters are each other's reverse — both
           readable, because the palindrome they were cut from was scored
           as a whole

and any arrangement  L1 L2 ... Lk  C  Rk ... R2 R1  of them is a palindrome by
construction, at ANY length. No search, no overhang, no letter debt: the
assembly step cannot fail, so the only open problem left at paragraph scale is
which units to choose and how to order them — and that is a problem about
meaning, not letters, which is the first time in this project the two have
separated.

The exhaustive hunt is what makes this more than a trick: it mass-produces the
units. 2,550,331 short palindromes from one 30-minute walk, each one either a
center or, when its mirror point lands on a word boundary, a ready-made pair.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from .validator import is_palindrome, normalize

_KNOWN: Optional[frozenset] = None


def is_novel_palindrome(text: str) -> bool:
    """Is this absent from the catalogued palindrome record?

    Checking a find against Brown and WikiText proves nothing — a known
    palindrome would not appear in either. `data/known_palindromes.json` holds
    the catalogued phrases, and any candidate has to clear it before being
    called new.
    """
    global _KNOWN
    if _KNOWN is None:
        import json
        from pathlib import Path
        try:
            _KNOWN = frozenset(json.loads(
                Path("data/known_palindromes.json").read_text()))
        except FileNotFoundError:
            _KNOWN = frozenset()
    return normalize(text) not in _KNOWN


@dataclass
class Bank:
    pairs: list[tuple[list[str], list[str]]] = field(default_factory=list)
    centers: list[list[str]] = field(default_factory=list)


def harvest(palindromes: Sequence[str]) -> Bank:
    """Sort found palindromes into assembly units.

    A palindrome whose mirror point falls in the gap between two words splits
    into a mirror-pair; one whose mirror runs through a word is indivisible
    and serves as a center.
    """
    bank = Bank()
    for text in palindromes:
        words = text.split()
        if not words or not is_palindrome(text):
            continue
        total = sum(len(w) for w in words)
        half, run, cut = total / 2, 0, None
        for i, w in enumerate(words):
            run += len(w)
            if run == half:
                cut = i + 1
                break
            if run > half:
                break
        if cut is not None:
            bank.pairs.append((words[:cut], words[cut:]))
        else:
            bank.centers.append(words)
    return bank


def assemble(pairs: Sequence[tuple[Sequence[str], Sequence[str]]],
             center: Optional[Sequence[str]]) -> list[str]:
    """Nest mirror-pairs around a center. Palindromic by construction.

    The left halves run in the given order; the right halves must close in
    the REVERSE order, exactly like brackets.
    """
    if not pairs and not center:
        return []
    out: list[str] = []
    for left, _ in pairs:
        out.extend(left)
    if center:
        out.extend(center)
    for _, right in reversed(pairs):
        out.extend(right)
    text = " ".join(out)
    assert is_palindrome(text), "assembly invariant broken"
    return out


def render(pairs: Sequence[tuple[Sequence[str], Sequence[str]]],
           center: Optional[Sequence[str]] = None,
           center_units: Optional[Sequence[str]] = None) -> str:
    """The paragraph as prose: one sentence per unit half.

    Punctuation and case are invisible to the mirror, so the paragraph can be
    typeset as ordinary sentences — the same license "A man, a plan, a canal:
    Panama" has always taken.
    """
    sentences: list[str] = []
    for left, _ in pairs:
        sentences.append(" ".join(left).capitalize() + ".")
    # A refrain core is many sentences. Passing it as one flat word list made
    # a 7-unit core render as a single 200-letter run-on.
    if center_units:
        sentences.extend(u.capitalize() + "." for u in center_units)
    elif center:
        sentences.append(" ".join(center).capitalize() + ".")
    for _, right in reversed(pairs):
        sentences.append(" ".join(right).capitalize() + ".")
    text = " ".join(sentences)
    # `assemble` asserts this; `render` did not, and silently emitted a
    # non-palindrome when given two centre units. A palindrome has one centre.
    assert is_palindrome(text), "render broke the mirror: check the centre"
    return text


def diversify(pairs, per_family: int = 3):
    """At most `per_family` pairs sharing a closing-half signature.

    The bank clusters: one strong mirror-core ("no it call...") spawns
    hundreds of pairs that read almost identically, and a selection menu
    drawn from the raw ranking hands the composer five of them.

    Which END clusters depends on where the pairs came from. Harvested pairs
    share a mirror-core, so siblings look alike closing. Mined pairs are keyed
    by an attested opening phrase, so "sites may", "sites but", "sites not" and
    "sites was" are siblings whose closing halves all differ. Both ends are
    capped, each by its leading word — the part siblings actually share, since
    "sites may" and "sites but" agree on nothing else. Coarse, and enough to
    stop a paragraph repeating itself.

    A pair and its flip are dropped to one outright, whatever the cap. When
    both halves are attested phrases, mining finds "not as || sat on" and "sat
    on || not as" alike; they are the same four words, and using both prints
    them twice.
    """
    from collections import Counter
    seen: Counter = Counter()
    used: set[frozenset] = set()
    out = []
    for left, right in pairs:
        material = frozenset((" ".join(left), " ".join(right)))
        if material in used:
            continue
        opening, closing = ("L",) + tuple(left[:1]), ("R",) + tuple(right[:1])
        if seen[opening] < per_family and seen[closing] < per_family:
            seen[opening] += 1
            seen[closing] += 1
            used.add(material)
            out.append((left, right))
    return out


def paragraph_words(pairs: Sequence[tuple[Sequence[str], Sequence[str]]],
                    center: Optional[Sequence[str]] = None) -> int:
    """How many words the assembled paragraph will hold.

    Both halves of every pair appear, which is why a 100-word paragraph needs
    only 50 words of chosen material — the mirror writes the rest.
    """
    return sum(len(l) + len(r) for l, r in pairs) + (len(center) if center else 0)


def enough_pairs(pairs: Sequence[tuple[Sequence[str], Sequence[str]]],
                 min_words: int = 100,
                 center: Optional[Sequence[str]] = None,
                 max_pairs: Optional[int] = None) -> list:
    """The shortest prefix of `pairs` that reaches the word floor.

    Criterion 1 is a floor, not a target: a paragraph that overshoots it is
    spending sentences a reader has to get through for nothing. `pairs` is
    expected to arrive best-first, so the prefix is also the best material.
    """
    taken: list = []
    for pair in pairs:
        if paragraph_words(taken, center) >= min_words:
            break
        if max_pairs is not None and len(taken) >= max_pairs:
            break
        taken.append((list(pair[0]), list(pair[1])))
    return taken


def _sequence(pairs: Sequence[tuple[Sequence[str], Sequence[str]]],
              center: Optional[Sequence[str]] = None) -> list[list[str]]:
    """The sentences in reading order, which is not the order of the pairs.

    Left halves run forwards and right halves close backwards, so pair k is
    read at position k and again at position 2k' - k. Any ordering decision has
    to be scored here rather than over the pairs, because moving one pair moves
    two sentences in opposite directions.
    """
    out = [list(l) for l, _ in pairs]
    if center:
        out.append(list(center))
    out += [list(r) for _, r in reversed(pairs)]
    return out


def adjacent_links(pairs: Sequence[tuple[Sequence[str], Sequence[str]]],
             center: Optional[Sequence[str]] = None) -> int:
    """Adjacent sentences sharing a content word, counted over the paragraph.

    A crude measure and a deliberately crude one. It is used to ORDER a set of
    sentences already chosen, never to choose them, and never reported as
    quality: four proxies in this project have disagreed with blind judging.
    Reordering cannot change what the paragraph is made of, so the worst this
    can do is pick a duller arrangement of the same material.
    """
    from .themes import content_words
    words = [content_words(" ".join(s)) for s in _sequence(pairs, center)]
    return sum(1 for a, b in zip(words, words[1:]) if a & b)


def order_pairs(pairs: Sequence[tuple[Sequence[str], Sequence[str]]],
                center: Optional[Sequence[str]] = None,
                rounds: int = 4) -> list:
    """Arrange the pairs so the paragraph reads with fewer jumps.

    A greedy pass over swaps. Small, because the objective is weak: it is worth
    taking the arrangement that repeats a word across a sentence break over the
    one that does not, and it is not worth searching hard for it.
    """
    best = [(list(l), list(r)) for l, r in pairs]
    score = adjacent_links(best, center)
    for _ in range(rounds):
        improved = False
        for i in range(len(best)):
            for j in range(i + 1, len(best)):
                trial = list(best)
                trial[i], trial[j] = trial[j], trial[i]
                gain = adjacent_links(trial, center)
                if gain > score:
                    best, score, improved = trial, gain, True
        if not improved:
            break
    return best


def refrain(units: Sequence[str]) -> list[str]:
    """A mirrored sequence of self-palindromic sentences: A B C ... C B A.

    Judged strictly, mirror-pairs with two readable halves barely exist — of
    the bank's top pairs, none passed on both sides. Whole short palindromes
    DO pass as sentences, and a mirrored sequence of them is a palindrome by
    construction. The repetition is not a defect; it is the refrain, and it is
    how every long palindromic poem already earns its length.
    """
    for u in units:
        if not is_palindrome(u):
            raise ValueError(f"not a palindrome: {u!r}")
    if len(units) == 1:
        return list(units)
    return list(units) + list(units[-2::-1])


def is_word_palindrome(text: str) -> bool:
    """Does the WORD sequence read the same both ways?

    A distinct constraint from the letter-level one, and a far cheaper one:
    it costs nothing per letter, so it survives at paragraph length where
    letter-level prose cannot. Popular palindrome lists mix the two — "Fall
    leaves as soon as leaves fall" is a word palindrome and not a letter one.
    """
    import re
    words = re.findall(r"[a-z]+", text.lower())
    return len(words) > 1 and words == words[::-1]


def word_assemble(outer: Sequence[str], center: str) -> str:
    """Bracket a word-palindromic center with whole sentences and their
    word-reversals.

    Splitting units at their midpoint left the paragraph's outer lines as
    fragments. A sentence's own word-reversal is a complete line, so the
    bracket keeps both halves whole:

        S1. S2. CENTER. reverse(S2). reverse(S1).
    """
    import re
    if not is_word_palindrome(center):
        raise ValueError(f"center is not word-palindromic: {center!r}")
    words = lambda s: re.findall(r"[A-Za-z]+", s)
    lines = [" ".join(words(s)).capitalize() + "." for s in outer]
    lines.append(" ".join(words(center)).capitalize() + ".")
    lines += [" ".join(reversed(words(s))).capitalize() + "."
              for s in reversed(outer)]
    return " ".join(lines)

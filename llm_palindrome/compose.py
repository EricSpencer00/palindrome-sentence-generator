"""Compose sentences instead of quoting them.

v2 puts judged-coherent English into a palindrome by placing whole Wikipedia
sentences whole. That works and it is quotation — every sentence a blinded
judge accepted was verbatim corpus text, and none of the composed ones passed.

This applies the project's own trick one level up. Palindromicity was never
something a model got right; it is something the search cannot violate, because
the trie only ever offers letter-valid continuations. Grammaticality can be the
same kind of guarantee. Take the part-of-speech skeleton of a sentence somebody
really wrote — DET NOUN VERB ADP DET NOUN — and fill each slot with a word the
bigram model likes. The result has the shape of English by construction and the
words are chosen, not copied.

Two filters then decide what survives, and the second is the one that matters:

  exclude    a composition that reproduces a corpus sentence is a quote again
  is_novel   and so is one that reproduces a corpus SPAN, which `exclude`
             cannot see
  mirror_ok  a sentence whose reversed letters cannot be spelled is useless
             here however well it reads

The second is why this is not simply a sentence generator. Roughly three in
four English sentences have no spellable mirror at a 30k vocabulary, so
composition has to be run against that filter rather than after it.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Callable, Iterable, Optional, Sequence


def pos_lexicon(tagged_sentences: Iterable[Sequence[tuple[str, str]]]
                ) -> dict[str, set[str]]:
    """word -> the tags it is attested with."""
    lex: dict[str, set[str]] = defaultdict(set)
    for sentence in tagged_sentences:
        for word, tag in sentence:
            lex[word.lower()].add(tag)
    return dict(lex)


PUNCT_TAGS = frozenset({".", "X"})
# Tags a clause can open on and still have a subject.
SUBJECT_TAGS = frozenset({"PRON", "DET", "NOUN", "ADJ", "NUM"})


def mine_templates(tagged_sentences: Iterable[Sequence[tuple[str, str]]],
                   min_words: int = 4, max_words: int = 8,
                   drop_tags: frozenset = PUNCT_TAGS,
                   sentence_shaped: bool = False) -> Counter:
    """The tag sequences of whole sentences, counted.

    Whole sentences, not spans: a template is meant to have a beginning and an
    end, which is exactly what mined n-grams lacked — "was unable to make" is
    a grammatical fragment and reads as one.
    """
    out: Counter = Counter()
    for sentence in tagged_sentences:
        # The full stop is a token in a tagged corpus. Keeping it makes the
        # template one slot too long and the composer fills that slot with a
        # word, which is where "manner as possible to" came from.
        shape = tuple(tag for _, tag in sentence if tag not in drop_tags)
        if not (min_words <= len(shape) <= max_words):
            continue
        # A tagged corpus calls headings and list items sentences too, so
        # mining by length alone yields ADP DET ADJ NOUN — "for more
        # information", a noun phrase that reads as a fragment.
        if sentence_shaped and not (
                "VERB" in shape and shape[0] in SUBJECT_TAGS):
            continue
        out[shape] += 1
    return out


def _by_tag(lexicon: dict[str, set[str]], vocab: set[str],
            rank: Optional[Callable[[str], float]] = None) -> dict[str, list[str]]:
    """Words available for each tag, best first.

    Ranked before the composer truncates the pool. Unranked, the pool came out
    in dictionary order and a cut at 300 meant the composer never saw "the".
    """
    table: dict[str, list[str]] = defaultdict(list)
    for word, tags in lexicon.items():
        if word in vocab:
            for tag in tags:
                table[tag].append(word)
    if rank is not None:
        for tag in table:
            table[tag].sort(key=rank)
    return dict(table)


def compose_sentences(templates: Counter, lexicon: dict[str, set[str]],
                      bigrams, vocab: set[str], n: int = 100,
                      exclude: Optional[set[str]] = None,
                      mirror_ok: Optional[Callable[[str], bool]] = None,
                      beam: int = 24,
                      max_candidates_per_slot: int = 400,
                      rank: Optional[Callable[[str], float]] = None,
                      unigram: Optional[Callable[[str], float]] = None,
                      is_novel: Optional[Callable[[str], bool]] = None) -> list[str]:
    """Fill attested templates with words the bigram model likes.

    Each template is filled left to right by a small beam over the words
    carrying the required tag, scored by the join with the word already
    placed. The beam is what makes this composition rather than sampling: the
    sentence is chosen for how it reads, subject to a shape English uses.
    """
    exclude = exclude or set()
    table = _by_tag(lexicon, vocab, rank=rank)
    out: list[str] = []
    seen: set[str] = set()

    for shape, _count in templates.most_common():
        if len(out) >= n:
            break
        if any(tag not in table for tag in shape):
            continue

        beams: list[tuple[float, list[str]]] = [(0.0, [])]
        for tag in shape:
            pool = table[tag][:max_candidates_per_slot]
            nxt: list[tuple[float, list[str]]] = []
            for score, words in beams:
                prev = words[-1] if words else None
                for w in pool:
                    if w in words:          # a sentence that repeats itself reads badly
                        continue
                    # The opening word has no join to be scored by, so without
                    # a unigram term the beam takes whatever heads the pool —
                    # which is how every sentence started with "ya" or "whoever".
                    joint = (bigrams.forward(prev, w) if prev
                             else (unigram(w) if unigram else 0.0))
                    nxt.append((score + joint, words + [w]))
            nxt.sort(key=lambda item: -item[0])
            beams = nxt[:beam]
            if not beams:
                break

        for _score, words in beams:
            sentence = " ".join(words)
            if len(words) != len(shape) or sentence in exclude or sentence in seen:
                continue
            if mirror_ok is not None and not mirror_ok("".join(words)):
                continue
            # `exclude` only holds whole corpus sentences. "it is a good idea"
            # is not a sentence in Brown but sits inside one, and reproducing it
            # is quotation however it was arrived at.
            if is_novel is not None and not is_novel(sentence):
                continue
            seen.add(sentence)
            out.append(sentence)
            if len(out) >= n:
                break
    return out[:n]

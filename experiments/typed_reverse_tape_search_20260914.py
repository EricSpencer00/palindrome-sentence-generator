"""Search authored clause space with a typed reverse-tape index.

The generator writes ordinary left clauses from a finite lexical grammar.  It
then reverses the *letters* and uses a trie dynamic program to enumerate word
segmentations on the other side.  No corpus sentence is copied and no model
score is queried per candidate.  A result is only a mechanical closure; the
rendered surfaces still require human readability evidence.
"""
from __future__ import annotations

import argparse
import heapq
import json
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from wordfreq import top_n_list, zipf_frequency

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_palindrome.safe_vocab import safe_vocab
from llm_palindrome.validator import is_palindrome, normalize


class Trie:
    def __init__(self, words: Iterable[str]):
        self.children: dict[str, dict] = {}
        self.words: set[str] = set()
        for word in words:
            self.words.add(word)
            node = self.children
            for char in word:
                node = node.setdefault(char, {})
            node.setdefault("", set()).add(word)

    def matches(self, text: str, start: int):
        node = self.children
        for pos in range(start, len(text)):
            node = node.get(text[pos])
            if node is None:
                return
            for word in node.get("", ()):
                yield pos + 1, word


@dataclass(frozen=True)
class LexicalGrammar:
    subjects: tuple[str, ...]
    verbs: tuple[str, ...]
    objects: tuple[str, ...]
    adjectives: tuple[str, ...]

    def clauses(self) -> Iterable[str]:
        """Yield complete, newly composed clauses in stable order."""
        for subject in self.subjects:
            for verb in self.verbs:
                for obj in self.objects:
                    yield f"{subject} {verb} {obj}"
            for copula in ("is", "was", "are", "were"):
                for adjective in self.adjectives:
                    yield f"{subject} {copula} {adjective}"


def default_grammar() -> LexicalGrammar:
    # These are lexical choices, not corpus spans.  The product is deliberately
    # auditable and can be frozen in an experiment manifest.
    nouns = (
        "artist author baker child doctor farmer friend gardener neighbor nurse"
        " parent sailor teacher worker writer dog cat bird river house garden"
        " letter story music idea plan map note book school town world day night"
        " rain wind fire water light song"
    ).split()
    subjects = tuple(
        ["i", "we", "you", "he", "she", "they"]
        + [f"{det} {noun}" for det in ("a", "the", "this", "that", "my", "our")
           for noun in nouns]
    )
    verbs = (
        "admired answered baked built called carried changed cleaned closed"
        " found fixed followed helped held kept learned liked listened loved"
        " made marked noticed opened painted planned read rescued saved saw sent"
        " showed studied taught thanked told used watched wrote"
    ).split()
    objects = tuple(
        [f"{det} {noun}" for det in ("a", "the", "this", "that", "my", "our")
         for noun in nouns]
        + [f"{det} {adj} {noun}"
           for det in ("a", "the", "this", "that")
           for adj in "old young kind quiet small bright dark clear open warm cold good true".split()
           for noun in nouns]
    )
    adjectives = tuple(
        "old young kind quiet brave calm small great new bright dark clear open"
        " warm cold long short good wise true ready safe gentle patient useful"
    .split())
    return LexicalGrammar(tuple(subjects), tuple(verbs), objects, adjectives)


def build_word_trie(limit: int = 50000, min_zipf: float = 3.0) -> tuple[Trie, dict[str, float]]:
    words = [
        word for word in top_n_list("en", limit)
        if word.isascii() and word.isalpha() and len(word) >= 2
        and zipf_frequency(word, "en") >= min_zipf
    ]
    words = safe_vocab(words)
    words = [
        word for word in words
        if len(word) <= 15 and not (len(word) >= 2 and len(set(word)) == 1)
    ] + ["a", "i"]
    words = list(dict.fromkeys(words))
    return Trie(words), {word: zipf_frequency(word, "en") for word in words}


def segment_reverse(text: str, trie: Trie, zipfs: dict[str, float], *,
                    bigrams: Counter | None = None,
                    followers: Counter | None = None,
                    top_k: int = 8, max_words: int = 12) -> list[tuple[float, tuple[str, ...]]]:
    """Return the best word-break paths for a reversed tape.

    The score is only a deterministic search ordering (frequency plus a mild
    length reward); it is explicitly not a readability certificate.
    """
    n = len(text)
    paths: list[list[tuple[float, tuple[str, ...]]]] = [[] for _ in range(n + 1)]
    paths[0] = [(0.0, ())]
    for start in range(n):
        if not paths[start]:
            continue
        for end, word in trie.matches(text, start):
            if len(paths[start][0][1]) >= max_words:
                continue
            additions = []
            for score, path in paths[start]:
                # Short words are legal English but are also the easiest way
                # for a word-break solver to manufacture noise.  Penalize
                # them in the traversal and apply the hard cap below; this is
                # a search-space guard, not a readability claim.
                short_penalty = 1.6 if len(word) <= 2 else 0.0
                if path and bigrams is not None and followers is not None:
                    # Corpus transitions are a deterministic traversal prior;
                    # they do not certify that the finished surface is prose.
                    seen = bigrams[(path[-1], word)]
                    if not seen:
                        continue
                    transition = 2.5 + 0.15 * seen.bit_length()
                else:
                    transition = 0.0
                additions.append((score + zipfs[word] * 0.7 + 0.12 * len(word)
                                  - short_penalty + transition, path + (word,)))
            paths[end].extend(additions)
            if len(paths[end]) > top_k * 4:
                paths[end] = heapq.nlargest(top_k * 2, paths[end], key=lambda row: row[0])
    return heapq.nlargest(top_k, paths[n], key=lambda row: row[0])


def run(*, max_clauses: int = 200_000, min_half_letters: int = 20,
        max_half_letters: int = 55, seed: int = 0, top_k: int = 64) -> dict:
    grammar = default_grammar()
    trie, zipfs = build_word_trie()
    bigrams: Counter | None = None
    followers: Counter | None = None
    # Brown is used only as a frozen transition table for word-break order.
    # The experiment never treats its score as a readability decision.
    try:
        from nltk.corpus import brown
        bigrams, followers = Counter(), Counter()
        for sentence in brown.sents():
            words = [word.casefold() for word in sentence if word.isalpha()]
            for left, right in zip(words, words[1:]):
                bigrams[(left, right)] += 1
                followers[left] += 1
    except LookupError:
        bigrams = followers = None
    clauses: list[str] = []
    seen: set[str] = set()
    for clause in grammar.clauses():
        tape = normalize(clause)
        if not (min_half_letters <= len(tape) <= max_half_letters):
            continue
        if clause in seen:
            continue
        seen.add(clause)
        clauses.append(clause)
        if len(clauses) >= max_clauses:
            break
    # Stable seeded order gives independent replay shards without using a
    # model to choose which lineage survives.
    rng = random.Random(seed)
    rng.shuffle(clauses)
    rows = []
    for left in clauses:
        left_tape = normalize(left)
        for score, right_words in segment_reverse(
                left_tape[::-1], trie, zipfs, bigrams=bigrams,
                followers=followers, top_k=top_k):
            if len(right_words) < 3:
                continue
            right = " ".join(right_words)
            if sum(len(word) <= 2 for word in right_words) > 2:
                continue
            text = f"{left} {right}"
            if left == right or not is_palindrome(text):
                continue
            rows.append({
                "text": text,
                "left": left,
                "right": right,
                "letters": len(normalize(text)),
                "search_score": score,
                "provenance": {
                    "generator": "typed_reverse_tape_search_20260914",
                    "left_source": "finite authored lexical product",
                    "right_source": "trie word-break over frozen top-word inventory",
                    "seed": seed,
                },
                "exact_palindrome": True,
            })
    rows.sort(key=lambda row: (-row["letters"], -row["search_score"], row["text"]))
    return {
        "status": "exact_closures_need_blinded_readability_evidence",
        "config": {
            "max_clauses": max_clauses,
            "min_half_letters": min_half_letters,
            "max_half_letters": max_half_letters,
            "seed": seed,
            "top_k": top_k,
        },
        "generated_clauses": len(clauses),
        "candidate_count": len(rows),
        "candidates": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-clauses", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    result = run(max_clauses=args.max_clauses, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("generated_clauses", "candidate_count")}, indent=2))


if __name__ == "__main__":
    main()

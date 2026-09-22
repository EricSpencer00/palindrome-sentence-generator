"""Live-overhang search over Brown-derived forward phrase units.

The phrase inventory is mined only as ordinary forward n-grams.  The search
does not reverse a completed sentence: it grows the two reading-order sides
outward while the character debt is consumed online.  This is a diagnostic
lane for finding a fresh intact candidate, not a readability certificate.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from collections import Counter
from pathlib import Path

from llm_palindrome.centerout import centerout_search
from llm_palindrome.search import WordTries, unit_letters


def letters(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())


def audit(text: str) -> dict:
    tape = letters(text)
    mismatch = next(((i, tape[i], tape[-i - 1])
                     for i in range(len(tape) // 2)
                     if tape[i] != tape[-i - 1]), None)
    return {"letters": len(tape), "exact": bool(tape) and mismatch is None,
            "first_mismatch": mismatch,
            "sha256_forward": hashlib.sha256(tape.encode()).hexdigest(),
            "sha256_reverse": hashlib.sha256(tape[::-1].encode()).hexdigest()}


class BrownPhraseScorer:
    wants_overhang = False

    def __init__(self, unigrams: Counter, bigrams: Counter, phrases: Counter):
        self.uni = unigrams
        self.bg = bigrams
        self.phrases = phrases
        self.total = max(1, sum(unigrams.values()))

    def _word(self, word: str) -> float:
        return math.log((self.uni.get(word, 0) + 0.2) / self.total)

    def _join(self, prev: str | None, word: str) -> float:
        if prev is None:
            return self._word(word)
        return math.log((self.bg.get((prev, word), 0) + 0.2)
                        / (self.uni.get(prev, 0) + 0.2 * len(self.uni)))

    def word_delta(self, left, right, placement, word, growth):
        words = word.split()
        score = 0.0
        if growth == "prepend":
            neighbor = left[0].split()[0] if left else None
            if neighbor:
                score += self._join(words[-1], neighbor)
            else:
                score += self._word(words[0])
        else:
            neighbor = right[-1].split()[-1] if right else None
            score += self._join(neighbor, words[0])
        score += sum(self._join(a, b) for a, b in zip(words, words[1:]))
        score += math.log1p(self.phrases.get(tuple(words), 0))
        score += 0.04 * len(unit_letters(word))
        return score


FUNCTION_WORDS = {
    "a", "an", "the", "as", "at", "by", "for", "from", "if", "in",
    "into", "is", "it", "of", "on", "or", "that", "to", "was", "were",
    "with", "and", "but", "be", "been", "are", "this", "these", "those",
}


def phrase_unit_allowed(unit: str) -> bool:
    words = unit.split()
    return any(word not in FUNCTION_WORDS and len(word) > 2 for word in words)


def state_allowed(left: tuple[str, ...], right: tuple[str, ...]) -> bool:
    units = left + right
    if len(units) != len(set(units)):
        return False
    words = " ".join(units).split()
    if any(a == b for a, b in zip(words, words[1:])):
        return False
    content = [word for word in words if word not in FUNCTION_WORDS]
    return len(content) == len(set(content))


def extract_brown(root: Path, max_units: int = 120_000,
                  min_n: int = 2, max_n: int = 4,
                  phrase_only: bool = False):
    unigram: Counter = Counter()
    bigram: Counter = Counter()
    phrase_counts: Counter = Counter()
    for path in sorted(root.iterdir()):
        if not path.is_file() or path.name in {"README", "CONTENTS"}:
            continue
        for raw_line in path.read_text(errors="ignore").splitlines():
            words = []
            for raw in raw_line.split():
                word = raw.rsplit("/", 1)[0].casefold()
                if re.fullmatch(r"[a-z]+", word):
                    words.append(word)
            unigram.update(words)
            bigram.update(zip(words, words[1:]))
            for n in range(min_n, max_n + 1):
                phrase_counts.update(tuple(words[i:i + n])
                                     for i in range(len(words) - n + 1))
    ranked = sorted(phrase_counts.items(),
                    key=lambda item: (-item[1], item[0]))[:max_units]
    # Preserve common single words too, but keep phrase units large enough to
    # carry real syntax; rare Brown tags and one-letter debris are excluded.
    singles = [word for word, count in unigram.most_common()
               if len(word) > 1 or word in {"a", "i"}][:20_000]
    units = ([] if phrase_only else singles) + [" ".join(words) for words, count in ranked]
    units = list(dict.fromkeys(units))
    return units, unigram, bigram, phrase_counts


def hidden_span(text: str) -> bool:
    words = re.findall(r"[a-z]+", text.casefold())
    full = letters(text)
    for i in range(len(words)):
        for j in range(i + 2, len(words) + 1):
            span = "".join(words[i:j])
            if 1 < len(span) < len(full) and span == span[::-1]:
                return True
    return False


def run(brown_root: Path, seconds: float = 45.0) -> dict:
    min_n = int(os.environ.get("BROWN_MIN_N", "2"))
    max_n = int(os.environ.get("BROWN_MAX_N", "4"))
    phrase_only = bool(os.environ.get("BROWN_PHRASE_ONLY"))
    units, unigram, bigram, phrases = extract_brown(
        brown_root,
        max_units=int(os.environ.get("BROWN_MAX_UNITS", "120000")),
        min_n=min_n,
        max_n=max_n,
        phrase_only=phrase_only)
    tries = WordTries(units)
    scorer = BrownPhraseScorer(unigram, bigram, phrases)
    deadline = time.monotonic() + seconds
    rows = []
    for seed in range(120):
        if time.monotonic() >= deadline:
            break
        words = centerout_search(
            tries, scorer, min_letters=39, beam_width=1200, max_steps=80,
            candidate_limit=500, seed=seed, diversity=0.35,
            max_overhang=18, deadline=deadline, maximize="letters",
            allow_word=lambda _placement, word, _state: phrase_unit_allowed(word),
            allow_state=state_allowed)
        if not words:
            continue
        text = " ".join(words)
        row = {"seed": seed, "rendered": text, "audit": audit(text),
               "hidden_palindromic_span": hidden_span(text),
               "units": words}
        if row["audit"]["exact"]:
            rows.append(row)
    rows.sort(key=lambda row: (-row["audit"]["letters"], row["rendered"]))
    return {"experiment": "brown-phrase-live-overhang-20260920",
            "method": "Brown forward phrase units with center-out live character debt",
            "stats": {"units": len(units), "seeds": 120,
                      "min_n": min_n, "max_n": max_n,
                      "phrase_only": phrase_only,
                      "exact": len(rows),
                      "longest_exact": max((r["audit"]["letters"] for r in rows), default=0),
                      "longest_clean_exact": max((r["audit"]["letters"] for r in rows
                                                   if not r["hidden_palindromic_span"]), default=0)},
            "exact_candidates": rows[:200],
            "provenance": {"source": "Brown POS-tagged forward n-grams",
                           "finished_tape_reversal": False,
                           "post_hoc_repair": False,
                           "audits": ["two-pointer mismatch", "forward/reverse SHA-256"],
                           "reader_gate": "closed pending human reading"}}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("brown_root", type=Path)
    parser.add_argument("--seconds", type=float, default=45.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.brown_root, args.seconds)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"]))
    for row in result["exact_candidates"][:20]:
        print(row["audit"]["letters"], row["hidden_palindromic_span"], row["rendered"])

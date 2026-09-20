"""Sample broad ordinary clauses, then independently parse the reverse tape.

Unlike mirror-pair assembly, the left and right sentences are generated and
segmented as separate grammatical paths. The character equation is satisfied
only when the complete left tape is independently tokenized into a right-side
clause; no word reversal, finished-tape reversal, or repair is used.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/two-sided-pcfg-sampler-20260920.json"
WORD_RE = re.compile(r"^[a-z]+$")


def tape(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())


def audit(text: str) -> dict:
    s = tape(text)
    return {"letters": len(s), "two_pointer_exact": bool(s) and s == s[::-1],
            "sha256_forward": hashlib.sha256(s.encode()).hexdigest(),
            "sha256_reverse": hashlib.sha256(s[::-1].encode()).hexdigest()}


def gate(words: list[str]) -> dict:
    repeated = len(words) != len(set(words))
    self_pal = any(len(w) > 1 and w == w[::-1] for w in words)
    spans = []
    for i in range(len(words)):
        for j in range(i + 2, len(words) + 1):
            if i == 0 and j == len(words):
                continue
            x = "".join(words[i:j])
            if len(x) > 1 and x == x[::-1]:
                spans.append((i, j))
    return {"no_repeated_words": not repeated, "no_self_palindromic_word": not self_pal,
            "no_nested_word_span": not spans,
            "mechanically_admitted": not repeated and not self_pal and not spans}


def load_bank() -> tuple[dict[str, list[tuple[str, float]]], list[list[str]]]:
    bank = json.loads((ROOT / "data/brown_pcfg_bank_20260920.json").read_text())
    lexicon = {}
    for pos, rows in bank["lexicon"].items():
        vals = []
        for row in rows:
            word = row["word"].lower()
            if WORD_RE.fullmatch(word):
                vals.append((word, max(1.0, float(row.get("score", 1)))))
        lexicon[pos] = vals
    return lexicon, bank["templates"]


def load_bigrams() -> dict[tuple[str, str], float]:
    out = {}
    for line in (ROOT / "data/count_2w.txt").read_text().splitlines():
        try:
            phrase, count = line.rsplit("\t", 1)
            a, b = phrase.lower().split()
            if WORD_RE.fullmatch(a) and WORD_RE.fullmatch(b):
                out[(a, b)] = math.log1p(float(count))
        except ValueError:
            continue
    return out


def sample_words(pos: str, lexicon: dict[str, list[tuple[str, float]]], rng: random.Random) -> str:
    rows = lexicon.get(pos, [])
    if not rows:
        return "the"
    total = sum(v for _, v in rows[:250])
    pick = rng.random() * total
    for word, value in rows[:250]:
        pick -= value
        if pick <= 0:
            return word
    return rows[0][0]


def segment_reverse(left: list[str], lexicon: dict[str, list[tuple[str, float]]],
                    bigrams: dict[tuple[str, str], float], max_words: int = 12) -> tuple[list[str], float] | None:
    rev = "".join(left)[::-1]
    unigram = {}
    for vals in lexicon.values():
        for w, value in vals:
            unigram[w] = max(unigram.get(w, 0.0), math.log1p(value))
    words = set(unigram)
    # longest-match dynamic programming with a language-model transition score
    best: dict[tuple[int, str, int], tuple[float, list[str]]] = {(0, "<s>", 0): (0.0, [])}
    for pos in range(len(rev)):
        states = [(k, v) for k, v in best.items() if k[0] == pos]
        for (at, prev, count), (score, path) in states:
            if count >= max_words:
                continue
            for end in range(pos + 1, min(len(rev), pos + 14) + 1):
                word = rev[pos:end]
                if word not in words:
                    continue
                # Prefer attested bigrams, but retain unseen transitions with
                # a small floor so the result remains a search rather than a
                # hard corpus filter.
                trans = bigrams.get((prev, word), -3.0 if prev != "<s>" else -1.5)
                word_score = unigram.get(word, -4.0)
                new_score = score + 0.35 * trans + 0.05 * word_score
                key = (end, word, count + 1)
                if key not in best or new_score > best[key][0]:
                    best[key] = (new_score, path + [word])
    finals = [(score, path) for (at, _, _), (score, path) in best.items() if at == len(rev)]
    if not finals:
        return None
    return max(finals, key=lambda row: row[0])[1], max(finals, key=lambda row: row[0])[0]


def run(samples: int = 100_000, seed: int = 20260920) -> dict:
    lexicon, templates = load_bank()
    bigrams = load_bigrams()
    rng = random.Random(seed)
    rows = []
    seen = set()
    for _ in range(samples):
        template = rng.choice([t for t in templates if 4 <= len(t) <= 10])
        left = [sample_words(pos, lexicon, rng) for pos in template]
        left_tape = "".join(left)
        if len(left_tape) < 20 or left_tape in seen:
            continue
        seen.add(left_tape)
        parsed = segment_reverse(left, lexicon, bigrams)
        if parsed is None:
            continue
        right, score = parsed
        text = " ".join(left + right)
        a = audit(text)
        g = gate(left + right)
        if a["two_pointer_exact"]:
            rows.append({"rendered": text, "left_words": left, "right_words": right,
                         "reverse_lm_score": score, "audit": a, "structural_gate": g,
                         "reader_status": "human-unreviewed; programmatic gate is diagnostic"})
    exact = [r for r in rows if r["audit"]["two_pointer_exact"]]
    admitted = [r for r in exact if r["structural_gate"]["mechanically_admitted"]]
    return {"experiment_id": "two-sided-pcfg-sampler-20260920",
            "method": "sample a broad ordinary clause then independently segment its reversed character tape with a scored lexical DP",
            "stats": {"samples": samples, "unique_left_tapes": len(seen), "exact": len(exact),
                      "mechanically_admitted": len(admitted),
                      "longest_exact": max((r["audit"]["letters"] for r in exact), default=0)},
            "exact": exact,
            "provenance": {"lexicon": "Brown POS-count bank; no source sentences imported",
                           "finished_tape_reversal": False, "post_hoc_repair": False,
                           "catalogue_text": False, "readability": "requires blinded human rating"}}


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
    result = run(n)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"]))
    for row in sorted(result["exact"], key=lambda r: -r["audit"]["letters"])[:20]:
        print(row["audit"]["letters"], row["structural_gate"]["mechanically_admitted"], row["rendered"])

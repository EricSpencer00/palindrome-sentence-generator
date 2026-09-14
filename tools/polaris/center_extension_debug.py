"""Polaris debug run for extending a verified readable palindrome.

The center is fixed to a human-selected palindrome and the search grows a new
left/right context around it. This is a constructive extension test, not a
catalogue walk: center words cannot be reused, self-palindromic units and
single-letter fillers are rejected, and every returned surface is audited by a
second exact tape check.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from llm_palindrome.bigram import BigramModel
from llm_palindrome.scoring import adjacent, first_word, last_word
from llm_palindrome.centerout import centerout_search
from llm_palindrome.search import WordTries, unit_letters
from llm_palindrome.validator import is_palindrome, normalize


CENTER = "an aide rips nine memos some men inspire diana"


class FrozenCoherentScorer:
    """Dependency-free local-order scorer for the frozen Polaris payload."""

    def __init__(self, words, bigrams, center):
        self.rank = {word: index for index, word in enumerate(words)}
        self.bg = bigrams
        self.center = center

    def word_delta(self, left, right, placement, word, growth):
        inner = word.split()
        rank_term = sum(8.0 - math.log1p(self.rank.get(part, len(self.rank)))
                        for part in inner)
        neighbor = adjacent(left, right, placement, growth)
        if neighbor is None:
            neighbor = self.center
        if growth == "prepend":
            order = self.bg.backward_order_gain(last_word(word), first_word(neighbor))
        else:
            order = self.bg.forward_order_gain(last_word(neighbor), first_word(word))
        order += sum(self.bg.forward_order_gain(a, b) for a, b in zip(inner, inner[1:]))
        existing = list(left) + list(right)
        used = sum(existing.count(part) for part in inner)
        return rank_term + 0.25 * order + 0.10 * len(unit_letters(word)) - 2.0 * used


def rank_and_size(default_size: int) -> tuple[int, int]:
    for rank_name, size_name in (("PALS_RANKID", "PALS_LOCAL_SIZE"),
                                 ("PMI_RANK", "PMI_SIZE"),
                                 ("SLURM_PROCID", "SLURM_NTASKS")):
        if rank_name in os.environ:
            return int(os.environ[rank_name]), int(os.environ.get(size_name) or default_size)
    return 0, default_size


def exact_audit(text: str, center: str) -> dict[str, object]:
    tape = normalize(text)
    center_tape = normalize(center)
    return {
        "independent_exact": bool(tape) and tape == tape[::-1],
        "validator_exact": is_palindrome(text),
        "center_exact": center_tape == center_tape[::-1],
        "letters": len(tape),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vocab", type=int, default=30000)
    ap.add_argument("--candidate-limit", type=int, default=200)
    ap.add_argument("--beam", type=int, default=64)
    ap.add_argument("--per-parent", type=int, default=8)
    ap.add_argument("--min-letters", type=int, default=100)
    ap.add_argument("--max-steps", type=int, default=240)
    ap.add_argument("--seeds-per-rank", type=int, default=8)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--maximize", choices=("score", "letters"), default="score")
    ap.add_argument("--bigram-path", default=None,
                    help="override the frozen count_2w path for local smoke tests")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    rank, size = rank_and_size(args.shards)
    words = open(f"{HERE}/payload/vocab30k.txt").read().split()[:args.vocab]
    tries = WordTries(words)
    bigram_path = args.bigram_path or f"{HERE}/payload/count_2w.txt"
    bigrams = BigramModel.from_file(bigram_path, vocab=words)
    center_words = set(CENTER.split())

    def allow_word(placement, word, state):
        letters = unit_letters(word)
        if len(letters) <= 1 or letters == letters[::-1]:
            return False
        existing = list(state.left) + list(state.right) + list(center_words)
        return all(existing.count(part) == 0 for part in word.split())

    def allow_closed(left, right):
        return bool(left and right)

    rows = []
    t0 = time.time()
    for offset in range(args.seeds_per_rank):
        seed = rank * args.seeds_per_rank + offset
        scorer = FrozenCoherentScorer(words, bigrams, CENTER)
        units = centerout_search(
            tries, scorer, center=CENTER, min_letters=args.min_letters,
            beam_width=args.beam, per_parent=args.per_parent,
            candidate_limit=args.candidate_limit, max_steps=args.max_steps,
            seed=seed, diversity=0.55, max_overhang=24,
            maximize=args.maximize, allow_word=allow_word,
            allow_closed=allow_closed)
        text = " ".join(units)
        audit = exact_audit(text, CENTER) if units else {
            "independent_exact": False, "validator_exact": False,
            "center_exact": True, "letters": 0,
        }
        if units:
            assert audit["independent_exact"] and audit["validator_exact"], text
        marker = CENTER
        split_at = units.index(marker) if marker in units else len(units)
        left, right = units[:split_at], units[split_at + 1:] if marker in units else []
        rows.append({
            "rank": rank, "seed": seed, "maximize": args.maximize,
            "closed": bool(units), "text": text, "left": left, "right": right,
            "outer_letters": len(normalize(" ".join(left + right))),
            "extension_words": len(left) + len(right), "audit": audit,
        })

    os.makedirs(args.out_dir, exist_ok=True)
    summary = {
        "rank": rank, "shards": size, "vocab": args.vocab,
        "center": CENTER, "center_sha256": hashlib.sha256(CENTER.encode()).hexdigest(),
        "maximize": args.maximize, "runs": len(rows),
        "closed": sum(row["closed"] for row in rows),
        "valid": sum(row["audit"]["independent_exact"] for row in rows),
        "longest_letters": max((row["audit"]["letters"] for row in rows), default=0),
        "seconds": round(time.time() - t0, 2), "rows": rows,
        "reader_status": "not_run", "machine_readability_certification": False,
    }
    with open(f"{args.out_dir}/summary_r{rank:04d}.json", "w") as fh:
        json.dump(summary, fh)
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}), flush=True)


if __name__ == "__main__":
    main()

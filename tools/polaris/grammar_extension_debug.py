"""Search grammar-constrained outer clauses around the authored center.

This is the next construction method after the unconstrained center extension:
Brown contributes only universal-POS tag sequences and word tags; all words are
selected by the exact center-out solver.  A closure must fill a complete
subject/verb-shaped clause on each side, use no center/repeated/self-palindromic
unit, and pass an independent tape audit.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from llm_palindrome.centerout import centerout_search
from llm_palindrome.search import WordTries, unit_letters
from llm_palindrome.validator import is_palindrome, normalize
from tools.polaris.center_extension_debug import CENTER, FrozenCoherentScorer
from llm_palindrome.bigram import BigramModel


OPEN = {"DET", "PRON", "NOUN", "ADJ", "NUM", "ADV"}


def rank_and_size(default_size: int) -> tuple[int, int]:
    for rank_name, size_name in (("PALS_RANKID", "PALS_LOCAL_SIZE"),
                                 ("PMI_RANK", "PMI_SIZE"),
                                 ("SLURM_PROCID", "SLURM_NTASKS")):
        if rank_name in os.environ:
            return int(os.environ[rank_name]), int(os.environ.get(size_name) or default_size)
    return 0, default_size


def load_payload(path: str):
    with gzip.open(path, "rt") as fh:
        blob = json.load(fh)
    table = {word: set(tags) for word, tags in blob["table"].items()}
    shapes = []
    for raw in blob["shapes"]:
        shape = tuple(raw)
        if 3 <= len(shape) <= 7 and shape[0] in OPEN and "VERB" in shape:
            shapes.append(shape)
    return table, list(dict.fromkeys(shapes))


def audit(text: str, left_shape, right_shape) -> dict[str, object]:
    tape = normalize(text)
    return {
        "text": text,
        "letters": len(tape),
        "left_shape": list(left_shape),
        "right_shape": list(right_shape),
        "independent_exact": bool(tape) and tape == tape[::-1],
        "validator_exact": is_palindrome(text),
        "reader_status": "not_run",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vocab", type=int, default=30000)
    ap.add_argument("--shape-limit", type=int, default=48)
    ap.add_argument("--candidate-limit", type=int, default=160)
    ap.add_argument("--beam", type=int, default=24)
    ap.add_argument("--per-parent", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=32)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--bigram-path", default=None)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    rank, size = rank_and_size(args.shards)
    words = open(f"{HERE}/payload/vocab30k.txt").read().split()[:args.vocab]
    table, shapes = load_payload(f"{HERE}/payload/brown.json.gz")
    shapes = shapes[:args.shape_limit]
    pairs = [(left, right) for left in shapes for right in shapes]
    assigned = pairs[rank::size]
    tries = WordTries(words)
    bigrams = BigramModel.from_file(args.bigram_path or f"{HERE}/payload/count_2w.txt", vocab=words)
    center_words = set(CENTER.split())
    rows = []
    stats = {"shape_pairs": len(assigned), "closures": 0, "exact": 0}
    t0 = time.time()

    for left_shape, right_shape in assigned:
        def allow_word(placement, word, state):
            letters = unit_letters(word)
            if len(letters) <= 1 or letters == letters[::-1]:
                return False
            if any(part in center_words for part in word.split()):
                return False
            existing = list(state.left) + list(state.right)
            if any(existing.count(part) for part in word.split()):
                return False
            tags = table.get(word, set())
            if placement == "L":
                pos = len(left_shape) - 1 - len(state.left)
                return 0 <= pos < len(left_shape) and left_shape[pos] in tags
            pos = len(state.right)
            return 0 <= pos < len(right_shape) and right_shape[pos] in tags

        def allow_closed(left, right):
            return len(left) == len(left_shape) and len(right) == len(right_shape)

        units = centerout_search(
            tries, FrozenCoherentScorer(words, bigrams, CENTER), center=CENTER,
            min_letters=len(normalize(CENTER)) + 12,
            beam_width=args.beam, per_parent=args.per_parent,
            candidate_limit=args.candidate_limit, max_steps=args.max_steps,
            seed=rank * 10000 + len(rows), diversity=0.35,
            max_overhang=24, maximize="score", allow_word=allow_word,
            allow_closed=allow_closed)
        if not units:
            continue
        text = " ".join(units)
        row = audit(text, left_shape, right_shape)
        rows.append(row)
        stats["closures"] += 1
        stats["exact"] += int(row["independent_exact"] and row["validator_exact"])

    rows.sort(key=lambda row: (-row["letters"], row["text"]))
    os.makedirs(args.out_dir, exist_ok=True)
    summary = {
        "rank": rank, "shards": size, "vocab": args.vocab,
        "shape_count": len(shapes), "assigned_shape_pairs": len(assigned),
        "center": CENTER,
        "center_sha256": hashlib.sha256(CENTER.encode()).hexdigest(),
        "stats": stats, "seconds": round(time.time() - t0, 2),
        "rows": rows, "machine_readability_certification": False,
    }
    with open(f"{args.out_dir}/summary_r{rank:04d}.json", "w") as fh:
        json.dump(summary, fh)
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}), flush=True)


if __name__ == "__main__":
    main()

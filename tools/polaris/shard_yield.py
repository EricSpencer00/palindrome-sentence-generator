"""One shard of the vocabulary-versus-length yield sweep.

The question this job exists to settle: at 27 letters and above, is the yield of
readable palindromes limited by the length or by the vocabulary? Locally a
1,200-word vocabulary produced zero sentence-shaped results in 99,446 draws
while a 6,000-word one produced a hit in 10,806, and the 30,000-word vocabulary
that can express half the catalogued record branched too widely to walk at all.
That last cell is the one that matters and it needs cores rather than cleverness.

`enumerate_palindromes` shards on the opening unit, and ranks partition the
openings exactly, so N ranks cover disjoint subtrees with no coordination and
no duplicate work. Each rank writes its own JSONL file; nothing is shared.

Self-contained on purpose
-------------------------
The vocabulary and the Brown tag tables are read from files frozen on the
laptop rather than rebuilt from `wordfreq` and `nltk`, so the job depends on
the standard library and this repository only. Installing packages into a
compute-node environment is the kind of failure that wastes an allocation, and
neither table changes between runs.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from llm_palindrome.exhaustive import enumerate_palindromes
from llm_palindrome.search import WordTries
from llm_palindrome.validator import is_palindrome, normalize


def rank_and_size(default_size: int) -> tuple[int, int]:
    """Which shard am I? Polaris runs PALS; MPICH and Slurm are covered too."""
    for r, s in (("PALS_RANKID", "PALS_LOCAL_SIZE"),
                 ("PMI_RANK", "PMI_SIZE"),
                 ("SLURM_PROCID", "SLURM_NTASKS")):
        if r in os.environ:
            return int(os.environ[r]), int(os.environ.get(s) or default_size)
    return 0, default_size


def load_brown(path: str):
    with gzip.open(path, "rt") as fh:
        blob = json.load(fh)
    table = {w: frozenset(t) for w, t in blob["table"].items()}
    shapes = {tuple(s) for s in blob["shapes"]}
    trigrams = {tuple(k.split("|")): v for k, v in blob["trigrams"].items()}
    return table, shapes, trigrams


def readings(words, table, limit=20000):
    from itertools import product
    pools, total = [], 1
    for w in words:
        tags = table.get(w)
        if not tags:
            return []
        total *= len(tags)
        if total > limit:
            return []
        pools.append(sorted(tags))
    return [tuple(r) for r in product(*pools)]


OPENING = frozenset({"PRON", "DET", "NOUN", "ADJ", "NUM", "ADV"})


def sentence_like(words, table, shapes) -> bool:
    """A verb, a subject-shaped opening, and an attested whole-sentence shape,
    all in the SAME tag reading. Kept identical to llm_palindrome/syntax.py."""
    for r in readings(words, table):
        if r in shapes and "VERB" in r and r[0] in OPENING:
            return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vocab", type=int, required=True)
    ap.add_argument("--lo", type=int, required=True)
    ap.add_argument("--hi", type=int, required=True)
    ap.add_argument("--budget", type=int, default=200_000_000)
    ap.add_argument("--max-units", type=int, default=40)
    ap.add_argument("--seconds", type=float, default=2400.0,
                    help="wall-clock budget; the queue limit is the real cap")
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    rank, size = rank_and_size(args.shards)
    words = open(f"{HERE}/payload/vocab30k.txt").read().split()[:args.vocab]
    table, shapes, _ = load_brown(f"{HERE}/payload/brown.json.gz")
    tries = WordTries(words)

    os.makedirs(args.out_dir, exist_ok=True)
    path = f"{args.out_dir}/v{args.vocab}_L{args.lo}-{args.hi}_r{rank:04d}.jsonl"

    seen: set[str] = set()
    hits = 0
    t0 = time.time()
    deadline = t0 + args.seconds
    with open(path, "w") as fh:
        for units in enumerate_palindromes(
                tries, max_letters=args.hi, min_letters=args.lo,
                node_budget=args.budget, max_units=args.max_units,
                shard=rank, shards=size, deadline=deadline, shuffle_seed=rank):
            key = normalize(" ".join(units))
            if key in seen:
                continue
            seen.add(key)
            if 3 <= len(units) <= 9 and sentence_like(units, table, shapes):
                text = " ".join(units)
                # The validator is cheap and this is the one thing that must
                # never be wrong in an output file.
                assert is_palindrome(text), text
                hits += 1
                fh.write(json.dumps({"letters": len(key), "text": text}) + "\n")
                fh.flush()

    summary = {"rank": rank, "shards": size, "vocab": args.vocab,
               "lo": args.lo, "hi": args.hi, "distinct": len(seen),
               "hits": hits, "seconds": round(time.time() - t0, 1)}
    with open(f"{args.out_dir}/summary_v{args.vocab}_L{args.lo}-{args.hi}"
              f"_r{rank:04d}.json", "w") as fh:
        json.dump(summary, fh)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

"""Do multi-word chunks beat single words, once the exponents are known?

The measured scaling says they should. Job 7553051 gave throughput falling as
1/V in the size of the unit inventory --- exponent +1.02 and +1.05 over two
independent steps --- while yield falls exponentially in length, about 1.7x per
letter. Length enters through the number of PLACEMENTS the walk has to make,
so a unit that covers eight letters instead of three and a half reaches forty
letters in five placements rather than eleven.

That is the trade: pay linearly in inventory size, save exponentially in depth.
Both sides of it are measured rather than assumed, which is what makes the
prediction worth testing instead of arguing about.

The repository has tried phrases before and recorded them as a failure:
attested bigrams consumed atomically drop bigram coverage from 0.70 to 0.48,
because locking two words together costs more at the seams than the internal
join buys. That measurement was about coverage of the joins. This one is about
findings per core-second at a length where the walk is otherwise too deep to
reach, and the two can both be true.

What is compared
----------------
Three inventories at matched compute and matched letter bands:

    words       the frequency-ranked vocabulary alone
    chunks      the same words plus the most frequent attested bigrams
    chunks-only the bigrams with a small word set for closing

Reported per cell: distinct palindromes found, distinct CORES (clustered by
the middle letters, because one centre reworded fifty ways is one finding), and
the mean number of units a result used, which is the depth the chunking is
supposed to be buying down.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_palindrome.exhaustive import enumerate_palindromes
from llm_palindrome.generate import build_vocab
from llm_palindrome.phrases import build_inventory, build_units
from llm_palindrome.search import WordTries
from llm_palindrome.syntax import brown_tables, sentence_like
from llm_palindrome.validator import is_palindrome, normalize


def core_of(letters: str, k: int = 14) -> str:
    m = len(letters) // 2
    return letters[max(0, m - k // 2):m + k // 2]


def words_in_units(units: list[str]) -> list[str]:
    """Expand atomic search units before applying a word-level grammar test."""
    return [word for unit in units for word in unit.split()]


def run(tries: WordTries, lo: int, hi: int, seconds: float, table, shapes,
        seed: int = 0) -> dict:
    seen: set[str] = set()
    hits: list[str] = []
    depths: list[int] = []
    word_depths: list[int] = []
    phrase_units_in_hits = 0
    t0 = time.time()
    for units in enumerate_palindromes(
            tries, max_letters=hi, min_letters=lo, node_budget=10 ** 12,
            max_units=40, deadline=t0 + seconds, shuffle_seed=seed):
        key = normalize(" ".join(units))
        if key in seen:
            continue
        seen.add(key)
        depths.append(len(units))
        words = words_in_units(units)
        word_depths.append(len(words))
        # `sentence_like` looks up Brown tags one word at a time.  Passing
        # "new york" as one unit made every real phrase use an automatic miss.
        if 3 <= len(words) <= 9 and sentence_like(words, table, shapes):
            text = " ".join(units)
            assert is_palindrome(text), text
            hits.append(text)
            phrase_units_in_hits += sum(" " in unit for unit in units)
    secs = time.time() - t0
    cores = {core_of(normalize(h)) for h in hits}
    return {"draws": len(seen), "hits": len(hits), "cores": len(cores),
            "mean_units": (sum(depths) / len(depths)) if depths else 0.0,
            "mean_words": (sum(word_depths) / len(word_depths)) if word_depths else 0.0,
            "phrase_units_in_hits": phrase_units_in_hits,
            "seconds": round(secs, 1),
            "draws_per_sec": len(seen) / max(1e-9, secs),
            "cores_per_ksec": len(cores) / max(1e-9, secs) * 1000,
            "examples": hits[:6]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vocab", type=int, default=6000)
    ap.add_argument("--phrases", type=int, default=20000)
    ap.add_argument("--bands", nargs="+", default=["27-31", "32-36", "37-41"])
    ap.add_argument("--seconds", type=float, default=90.0,
                    help="wall-clock budget for each arm/band/seed cell")
    ap.add_argument("--seeds", type=int, default=1,
                    help="independent shuffled walks per arm/band (default: 1)")
    ap.add_argument("--out", default="experiments/chunk_scaling.json")
    args = ap.parse_args()

    words = build_vocab(30000)[:args.vocab]
    table, shapes, _ = brown_tables()
    inventory = build_inventory("data/count_2w.txt", words,
                                top_n=args.phrases, min_count=1)
    print(f"words {len(words)}, attested bigrams kept {len(inventory)}")

    arms = {
        "words": list(words),
        "chunks": build_units(words, inventory),
        "chunks-only": build_units(words[:800], inventory),
    }
    for name, units in arms.items():
        letters = [len(u.replace(" ", "")) for u in units]
        print(f"  {name:12s} {len(units):7d} units, "
              f"mean {sum(letters)/len(letters):.2f} letters each")

    rows = []
    print(f"\n{'arm':>12} {'band':>8} {'seed':>4} {'draws':>9} {'draws/s':>9} "
          f"{'units':>6} {'hits':>5} {'cores':>6} {'cores/ks':>9}")
    for band in args.bands:
        lo, hi = (int(x) for x in band.split("-"))
        for seed in range(args.seeds):
            for name, units in arms.items():
                tries = WordTries(units)
                r = run(tries, lo, hi, args.seconds, table, shapes, seed=seed)
                r.update(arm=name, lo=lo, hi=hi, seed=seed)
                rows.append(r)
                print(f"{name:>12} {band:>8} {seed:4d} {r['draws']:9,} "
                      f"{r['draws_per_sec']:9.1f} {r['mean_units']:6.2f} "
                      f"{r['hits']:5d} {r['cores']:6d} {r['cores_per_ksec']:9.2f}")
                for ex in r["examples"][:2]:
                    print(f"{'':>27} {len(normalize(ex)):2d}  {ex}")

    with open(args.out, "w") as fh:
        json.dump(rows, fh, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

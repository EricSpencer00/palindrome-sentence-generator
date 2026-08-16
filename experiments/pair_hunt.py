"""Generate mirror-pairs, then let a model choose among them.

The units this project shipped came from `data/known_palindromes.json` — the
catalogue. Criterion 9 of docs/NORTH-STAR.md says that does not count, and it
is the criterion the whole v2 paragraph fails. This walks the space instead.

Two stages, and the split matters. Enumeration is CPU-bound, embarrassingly
parallel, and produces far more than anything can read: a 90-second single
process run returns thousands of letter-valid pairs. Scoring is the scarce
part, so it happens once, in batch, over everything that survived the
structural filters.

The model here filters. It does not decide: four proxies in this project have
disagreed with blind judging and none has ever agreed on ranking, so the output
of this script is a shortlist for a person to read, not a bank to ship.

    python experiments/pair_hunt.py --time-budget 600 --workers 6
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

from llm_palindrome.generate import build_vocab
from llm_palindrome.lexicon import load_lexicon
from llm_palindrome.pairs import hunt, junction, pair_vocabulary
from llm_palindrome.paragraphs import is_novel_palindrome
from llm_palindrome.search import WordTries


def _zipf():
    from wordfreq import zipf_frequency
    cache: dict[str, float] = {}

    def zipf(word: str) -> float:
        hit = cache.get(word)
        if hit is None:
            hit = cache[word] = zipf_frequency(word, "en")
        return hit
    return zipf


def _worker(args) -> list[tuple[list[str], list[str]]]:
    (indices, shards, vocab_n, min_zipf, node_budget, min_letters, max_letters,
     max_overhang, max_units, min_words, per_family, deadline) = args
    zipf = _zipf()
    vocab = pair_vocabulary(build_vocab(vocab_n), zipf,
                            load_lexicon("data/lexicon.txt"), min_zipf)
    tries = WordTries(vocab)
    return list(hunt(tries, shards=shards, node_budget=node_budget,
                     min_letters=min_letters, max_letters=max_letters,
                     max_overhang=max_overhang, max_units=max_units,
                     min_words=min_words, per_family=per_family,
                     deadline=deadline, shard_indices=indices))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", type=int, default=2400,
                    help="openings the space is cut into; the walk takes a "
                         "small budget in each rather than exhausting one")
    ap.add_argument("--node-budget", type=int, default=60000)
    ap.add_argument("--min-letters", type=int, default=20)
    ap.add_argument("--max-letters", type=int, default=34)
    ap.add_argument("--max-overhang", type=int, default=16)
    ap.add_argument("--max-units", type=int, default=10)
    ap.add_argument("--min-words", type=int, default=3,
                    help="words per half; two-word halves are fragments")
    ap.add_argument("--per-family", type=int, default=3,
                    help="pairs sharing a junction — the mirror core siblings "
                         "are generated from")
    ap.add_argument("--min-zipf", type=float, default=3.6)
    ap.add_argument("--vocab", type=int, default=40000)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--time-budget", type=float, default=600.0)
    ap.add_argument("--model", default="gpt2", help="'' to skip scoring")
    ap.add_argument("--top", type=int, default=3000)
    ap.add_argument("--out", default="runs/pair_hunt.json")
    args = ap.parse_args()

    deadline = time.time() + args.time_budget
    jobs = [(list(range(i, args.shards, args.workers)), args.shards, args.vocab,
             args.min_zipf, args.node_budget, args.min_letters,
             args.max_letters, args.max_overhang, args.max_units,
             args.min_words, args.per_family, deadline)
            for i in range(args.workers)]

    t0 = time.time()
    if args.workers == 1:
        found = _worker(jobs[0])
    else:
        with mp.Pool(args.workers) as pool:
            found = [p for chunk in pool.map(_worker, jobs) for p in chunk]
    walked = time.time() - t0

    # Each process caps families against its own counter, so the cap has to be
    # reapplied over the union or a junction can appear `workers` times over.
    from collections import Counter
    seen: Counter = Counter()
    pairs = []
    for left, right in found:
        fam = junction(left, right)
        if seen[fam] >= args.per_family:
            continue
        seen[fam] += 1
        pairs.append((left, right))

    novel = [(l, r) for l, r in pairs
             if is_novel_palindrome(" ".join(l + r))]

    ranked = []
    if args.model and novel:
        from llm_palindrome.lm_scoring import GPT2Scorer
        lm = GPT2Scorer(args.model)
        texts = [" ".join(w).capitalize() + "." for pair in novel for w in pair]
        scores = lm.score_texts(texts, batch_size=64)
        for i, (left, right) in enumerate(novel):
            a, b = scores[2 * i], scores[2 * i + 1]
            # The weaker half is what a reader will trip on, so a pair is worth
            # its worse side rather than its average — an average lets one
            # fluent half carry a fragment.
            ranked.append({"left": left, "right": right,
                           "score": round(min(a, b), 4),
                           "mean": round((a + b) / 2, 4)})
        ranked.sort(key=lambda r: -r["score"])
    else:
        ranked = [{"left": l, "right": r} for l, r in novel]

    out = {"config": vars(args), "walked_seconds": round(walked, 1),
           "found": len(found), "deduped": len(pairs), "novel": len(novel),
           "results": ranked[:args.top]}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"walked {walked:.0f}s  found {len(found)}  deduped {len(pairs)}  "
          f"novel {len(novel)}  wrote {min(len(ranked), args.top)} -> {args.out}")
    for row in ranked[:25]:
        print(f"  {row.get('score', 0):7.3f}  {' '.join(row['left'])} || "
              f"{' '.join(row['right'])}")


if __name__ == "__main__":
    main()

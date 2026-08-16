"""Walk the space of short palindromes, then let a model choose among all of it.

The record says readable palindromes are short — 24 letters for "A man, a plan,
a canal: Panama", 30 for "Sir, I demand, I am a maid named Iris" — and that
nothing long is readable at all: the 90,439-letter record holder is a noun list
its author calls nonsense. Every search in this repository has been a beam
aimed at lengths where nothing good exists.

At 24 letters the space is walkable, which changes what the model is for.
`oracle_bound.py` showed reranking a beam's output is exhausted, and that bound
is over a fixed proposal distribution — an exhaustive walk has no proposal
distribution, so the bound does not apply to it. The model stops steering and
starts choosing.

Two stages, because they want different hardware:

  enumerate  CPU, embarrassingly parallel, sharded on the opening unit
  score      GPU, one batched pass over everything that survived filtering

    python experiments/exhaustive_hunt.py --max-letters 26 --workers 32
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

from llm_palindrome.exhaustive import (acceptable_words, enumerate_palindromes,
                                       hunt_vocabulary)
from llm_palindrome.generate import build_vocab
from llm_palindrome.search import WordTries
from llm_palindrome.shortwords import is_real_short
from llm_palindrome.validator import is_palindrome, normalize


def acceptable(words: list[str], zipf, min_zipf: float, min_distinct: int,
               min_mean_len: float) -> bool:
    """Reject before the GPU sees it.

    An exhaustive walk produces "a a a a a a" and every other degenerate
    closure the vocabulary allows, and scoring those is the only way this job
    could waste an allocation. The filters are the ones the record's readable
    palindromes would all pass.
    """
    if len(set(words)) < min_distinct:
        return False
    if any(a == b for a, b in zip(words, words[1:])):
        return False
    if not all(is_real_short(w) for w in words):
        return False
    if not acceptable_words(words, min_mean_len=min_mean_len):
        return False
    return all(zipf(w) >= min_zipf for w in words)


def _worker(args) -> list[list[str]]:
    (shard, shards, vocab_n, max_letters, min_letters, max_overhang,
     node_budget, max_units, min_zipf, min_distinct, min_mean_len,
     deadline, shuffle) = args
    from wordfreq import zipf_frequency

    cache: dict[str, float] = {}

    def zipf(w: str) -> float:
        hit = cache.get(w)
        if hit is None:
            hit = cache[w] = zipf_frequency(w, "en")
        return hit

    # Pruned before the trie is built, so the walk never spends a node on a
    # unit whose closures would all be rejected.
    vocab = hunt_vocabulary(build_vocab(vocab_n), zipf, min_zipf)
    tries = WordTries(vocab)

    out = []
    for units in enumerate_palindromes(tries, max_letters=max_letters,
                                       min_letters=min_letters,
                                       max_overhang=max_overhang,
                                       shard=shard, shards=shards,
                                       node_budget=node_budget,
                                       max_units=max_units,
                                       deadline=deadline,
                                       shuffle_seed=(shard if shuffle else None)):
        words = [w for u in units for w in u.split()]
        if acceptable(words, zipf, min_zipf, min_distinct, min_mean_len):
            out.append(words)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-letters", type=int, default=26)
    ap.add_argument("--min-letters", type=int, default=16)
    ap.add_argument("--max-overhang", type=int, default=18)
    ap.add_argument("--max-units", type=int, default=10)
    ap.add_argument("--vocab", type=int, default=60000)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--node-budget", type=int, default=10 ** 9)
    ap.add_argument("--time-budget", type=float, default=0.0,
                    help="seconds for the WALK; the stopping rule a walltime "
                         "queue actually needs. 40M nodes was ~2.8h per shard "
                         "against a 55-minute limit — both runs died with "
                         "nothing. 0 disables.")
    ap.add_argument("--min-zipf", type=float, default=3.5,
                    help="every word this common — the anti-gibberish filter")
    ap.add_argument("--min-distinct", type=int, default=4)
    ap.add_argument("--shuffle", action="store_true",
                    help="randomise the frontier. A LIFO walk under a time "
                         "budget drills one corner: 2.55M results contained "
                         "none of the 27 canonical palindromes.")
    ap.add_argument("--min-mean-len", type=float, default=3.2,
                    help="mean word length; keeps the walk off tiny filler")
    ap.add_argument("--top", type=int, default=400)
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="results/exhaustive_hunt.json")
    args = ap.parse_args()

    # Report the cores the job ACTUALLY has, from inside the process — the
    # only place it can be trusted. Both failed runs printed "1" from the batch
    # shell, which said nothing about what mpiexec provided.
    try:
        visible = len(os.sched_getaffinity(0))
    except AttributeError:      # macOS
        visible = os.cpu_count() or 1
    workers = min(args.workers, visible)
    print(f"visible cores: {visible}; using {workers} workers", flush=True)

    t0 = time.time()
    deadline = (time.time() + args.time_budget) if args.time_budget else None
    jobs = [(i, workers, args.vocab, args.max_letters, args.min_letters,
             args.max_overhang, args.node_budget, args.max_units,
             args.min_zipf, args.min_distinct, args.min_mean_len, deadline,
             args.shuffle)
            for i in range(workers)]

    # Shards land on disk as they finish. A queue job that is killed at
    # walltime otherwise loses everything it walked, which is what happened to
    # the first submission: it ran 55 minutes and wrote nothing, because the
    # results were still inside a Pool.map that never returned.
    partial = Path(args.out).with_suffix(".partial.jsonl")
    partial.parent.mkdir(parents=True, exist_ok=True)
    found: list[list[str]] = []
    with mp.Pool(workers) as pool, partial.open("w") as fh:
        for done, part in enumerate(pool.imap_unordered(_worker, jobs), 1):
            found.extend(part)
            for words in part:
                fh.write(json.dumps(words) + "\n")
            fh.flush()
            print(f"  shard {done}/{workers} done, {len(found):,} kept "
                  f"({time.time() - t0:.0f}s)", flush=True)
    walk = time.time() - t0

    seen, cands = set(), []
    for words in found:
        text = " ".join(words)
        if text in seen or not is_palindrome(text):
            continue
        seen.add(text)
        cands.append(text)
    print(f"walked in {walk:.0f}s: {len(found):,} accepted, {len(cands):,} distinct",
          flush=True)

    if not cands:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"config": vars(args), "results": []}))
        return

    import torch
    from llm_palindrome.lm_scoring import GPT2Scorer
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    lm = GPT2Scorer(args.model, device=device)
    print(f"scoring {len(cands):,} on {device}", flush=True)

    t1 = time.time()
    scored = []
    B = 256
    for i in range(0, len(cands), B):
        batch = cands[i:i + B]
        for text, s in zip(batch, lm.score_texts(
                [t.capitalize() + "." for t in batch], batch_size=B)):
            scored.append((s, text))
    scored.sort(reverse=True)
    print(f"scored in {time.time() - t1:.0f}s", flush=True)

    results = [{"score": round(s, 4), "letters": len(normalize(t)), "text": t}
               for s, t in scored[:args.top]]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"config": vars(args), "walked_seconds": round(walk, 1),
         "distinct": len(cands), "results": results}, indent=2))

    print(f"\nbest {min(25, len(results))}:")
    for r in results[:25]:
        print(f"  {r['score']:+.3f}  [{r['letters']:2d}]  {r['text']}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    main()

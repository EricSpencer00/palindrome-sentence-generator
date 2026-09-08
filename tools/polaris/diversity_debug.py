"""Hard-diversity versus semantic-proposal search benchmark."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from llm_palindrome.bigram import BigramModel
from llm_palindrome.overhang import DebtIndex, OverhangAware
from llm_palindrome.search import WordTries, beam_search
from llm_palindrome.semantic import RankOrderScorer
from llm_palindrome.validator import is_palindrome, normalize


def rank_and_size(default_size: int) -> tuple[int, int]:
    for rank_name, size_name in (("PALS_RANKID", "PALS_LOCAL_SIZE"),
                                 ("PMI_RANK", "PMI_SIZE"),
                                 ("SLURM_PROCID", "SLURM_NTASKS")):
        if rank_name in os.environ:
            return int(os.environ[rank_name]), int(os.environ.get(size_name) or default_size)
    return 0, default_size


def text_metrics(words: list[str], bigrams: BigramModel) -> dict:
    pairs = list(zip(words, words[1:]))
    gains = [bigrams.forward_order_gain(a, b) for a, b in pairs]
    counts = {word: words.count(word) for word in set(words)}
    return {
        "distinct_ratio": round(len(counts) / max(1, len(words)), 4),
        "max_word_uses": max(counts.values(), default=0),
        "attested_pair_rate": round(
            sum(bigrams.observed(a, b) for a, b in pairs) / max(1, len(pairs)), 4),
        "mean_order_gain": round(statistics.mean(gains), 4) if gains else 0.0,
        "length_template": "-".join(str(len(word)) for word in words),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vocab-file", default=f"{HERE}/payload/vocab30k.txt")
    ap.add_argument("--bigram-file", default=f"{HERE}/payload/count_2w.txt")
    ap.add_argument("--vocab", type=int, default=30000)
    ap.add_argument("--candidate-limit", type=int, default=200)
    ap.add_argument("--opening-pool", type=int, default=1024)
    ap.add_argument("--beam", type=int, default=64)
    ap.add_argument("--per-parent", type=int, default=8)
    ap.add_argument("--min-letters", type=int, default=80)
    ap.add_argument("--max-steps", type=int, default=260)
    ap.add_argument("--seeds-per-rank", type=int, default=4)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    rank, size = rank_and_size(args.shards)
    words = open(args.vocab_file).read().split()[:args.vocab]
    tries = WordTries(words)
    bigrams = BigramModel.from_file(args.bigram_file, vocab=words)
    # Round-robin partitions are disjoint across ranks and retain every length
    # band represented by the trie's balanced candidate proposal.
    opening_pool = tries.left_candidates("", args.opening_pool)
    rank_openings = opening_pool[rank::size]
    arms = (("baseline", 0.0, False),
            ("diverse", 0.0, True),
            ("diverse_sem", 0.25, True))
    rows = []
    t0 = time.time()
    for arm, weight, constrained in arms:
        used_openings: set[str] = set()
        for offset in range(args.seeds_per_rank):
            seed = rank * args.seeds_per_rank + offset
            base = RankOrderScorer(words, bigrams, order_weight=weight)
            scorer = OverhangAware(base, DebtIndex(tries, limit=96),
                                   debt_weight=2.0, dead_penalty=30.0)
            allowed = (set(rank_openings) - used_openings) if constrained else None
            units = beam_search(
                tries, scorer, min_letters=args.min_letters,
                beam_width=args.beam, per_parent=args.per_parent,
                candidate_limit=args.candidate_limit, max_steps=args.max_steps,
                seed=seed, opening_words=allowed,
                max_word_uses=2 if constrained else None)
            text = " ".join(units)
            valid = bool(units) and is_palindrome(text)
            if units:
                assert valid, text
                used_openings.add(units[0])
            row = {"rank": rank, "seed": seed, "arm": arm,
                   "order_weight": weight, "closed": bool(units), "valid": valid,
                   "letters": len(normalize(text)), "words": len(units),
                   "opening": units[0] if units else None, "text": text}
            if units:
                row.update(text_metrics(units, bigrams))
            rows.append(row)

    os.makedirs(args.out_dir, exist_ok=True)
    summary = {"rank": rank, "shards": size, "vocab": args.vocab,
               "opening_partition": rank_openings, "runs": len(rows),
               "closed": sum(row["closed"] for row in rows),
               "seconds": round(time.time() - t0, 2), "rows": rows}
    with open(f"{args.out_dir}/summary_r{rank:04d}.json", "w") as fh:
        json.dump(summary, fh)
    print(json.dumps({key: value for key, value in summary.items()
                      if key not in {"rows", "opening_partition"}}), flush=True)


if __name__ == "__main__":
    main()

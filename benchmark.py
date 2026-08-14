"""Compare scoring configurations at matched search budgets.

Reports length, sentence count, wall time, GPT-2 fluency, and an independent
palindrome check for each arm. Used for the results table in the writeup.
"""
from __future__ import annotations

import argparse
import json
import time

from llm_palindrome.generate import ZipfScorer, build_vocab, make_lm_prune
from llm_palindrome.lm_scoring import GPT2Scorer
from llm_palindrome.search import WordTries, beam_search
from llm_palindrome.textify import textify
from llm_palindrome.validator import is_palindrome, normalize


def run_arm(name, tries, scorer, lm, seeds, min_letters, beam, prune):
    t0 = time.time()
    cands = []
    for seed in range(seeds):
        w = beam_search(tries, scorer, min_letters=min_letters,
                        beam_width=beam, seed=seed, prune=prune)
        if w:
            cands.append(w)
    elapsed = time.time() - t0
    if not cands:
        return {"arm": name, "found": 0, "seconds": round(elapsed, 1)}
    texts = [textify(w) for w in cands]
    scores = lm.score_texts(texts)
    best_i = max(range(len(texts)), key=lambda i: scores[i])
    best = texts[best_i]
    return {
        "arm": name,
        "found": len(cands),
        "seconds": round(elapsed, 1),
        "letters": len(normalize(best)),
        "sentences": best.count("."),
        "lm_score": round(scores[best_i], 3),
        "mean_lm_score": round(sum(scores) / len(scores), 3),
        "valid_palindrome": is_palindrome(best),
        "all_valid": all(is_palindrome(t) for t in texts),
        "text": best,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--min-letters", type=int, default=200)
    ap.add_argument("--beam", type=int, default=60)
    ap.add_argument("--out", default="benchmark_results.json")
    args = ap.parse_args()

    tries = WordTries(build_vocab())
    scorer = ZipfScorer()
    lm = GPT2Scorer("gpt2")

    results = [
        run_arm("zipf+lm_rerank", tries, scorer, lm, args.seeds,
                args.min_letters, args.beam, None),
        run_arm("zipf+lm_in_loop", tries, scorer, lm, args.seeds,
                args.min_letters, args.beam,
                make_lm_prune(lm, lambda ws: " ".join(ws), keep=max(8, args.beam // 3))),
    ]

    for r in results:
        print(json.dumps({k: v for k, v in r.items() if k != "text"}, indent=None))
        if "text" in r:
            print(f"  {r['text'][:120]}...\n")
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

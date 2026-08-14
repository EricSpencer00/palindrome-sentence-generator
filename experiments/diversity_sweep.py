"""Does the search explore, and does exploring buy readability?

`oracle_bound.py` established that reranking the search's output is exhausted:
best-of-2000 scores 0.168 nats/token above best-of-24 and reads no better. But
that bound turned out to be a bound on one corridor rather than on the space.
Of 2000 palindromes, 1975 opened with the same five words and 1983 closed with
the same five, and the whole sample used 1317 of the 30000 available words.

The cause is the beam's only source of seed variation. `beam_search` adds
`rng.random() * diversity` to each candidate, with diversity=0.4, while
ZipfScorer's own range is several units wide — word frequency alone spans 0 to
8. The jitter cannot reorder the leading candidates, so every seed walks into
the same opening and only the interior varies.

This sweep raises that one number and measures two things that must be
separated:

  - **exploration**, which is certain to improve, since the jitter is what was
    suppressing it; and
  - **readability**, which is the actual question. A search that wanders more
    is not obviously a search that writes better, and it may close less often.

Reporting exploration alone would prove nothing about the goal, so the judge
score and the closure rate are carried alongside it. Judged per token by the
same model oracle_bound used, on the finished text in reading order.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.oracle_bound import load_scorer_model, score_texts
from llm_palindrome.generate import ZipfScorer, build_vocab
from llm_palindrome.search import WordTries, beam_search
from llm_palindrome.textify import textify
from llm_palindrome.validator import is_palindrome

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(REPO, "runs", "diversity_sweep.json")

DIVERSITIES = [0.4, 1.0, 2.0, 4.0, 6.0]


def opening(text: str, k: int = 5) -> str:
    return " ".join(text.replace(".", "").lower().split()[:k])


def ending(text: str, k: int = 5) -> str:
    return " ".join(text.replace(".", "").lower().split()[-k:])


def generate_arm(tries, scorer, diversity: float, seeds: int,
                 min_letters: int, beam_width: int) -> tuple[list[str], int]:
    texts: list[str] = []
    skipped = 0
    for seed in range(seeds):
        words = beam_search(tries, scorer, min_letters=min_letters,
                            beam_width=beam_width, seed=seed,
                            diversity=diversity)
        if not words:
            skipped += 1
            continue
        text = textify(words)
        assert is_palindrome(text), f"FATAL: diversity={diversity} seed={seed}"
        texts.append(text)
    return texts, skipped


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--min-letters", type=int, default=200)
    ap.add_argument("--beam", type=int, default=60)
    ap.add_argument("--vocab", type=int, default=30000)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--fallback-model", default="gpt2-large")
    ap.add_argument("--diversities", type=float, nargs="+", default=DIVERSITIES)
    ap.add_argument("--out", default=OUT_PATH)
    args = ap.parse_args()

    tries = WordTries(build_vocab(args.vocab))
    scorer = ZipfScorer()

    arms = []
    for div in args.diversities:
        t0 = time.time()
        texts, skipped = generate_arm(tries, scorer, div, args.seeds,
                                      args.min_letters, args.beam)
        print(f"[gen] diversity={div}: {len(texts)}/{args.seeds} closed, "
              f"{skipped} skipped, {time.time() - t0:.0f}s", flush=True)
        arms.append({"diversity": div, "texts": texts, "skipped": skipped})

    model_used, tok, model, device = load_scorer_model(args.model,
                                                       args.fallback_model)

    report = {"params": {"seeds": args.seeds, "min_letters": args.min_letters,
                         "beam": args.beam, "vocab": args.vocab},
              "judge_model": model_used, "arms": []}

    for arm in arms:
        texts = arm["texts"]
        if not texts:
            report["arms"].append({"diversity": arm["diversity"],
                                   "closed": 0, "skipped": arm["skipped"]})
            continue
        scores = score_texts(texts, tok, model, device)
        per_token = [s["per_token"] for s in scores]
        words = [t.replace(".", "").lower().split() for t in texts]
        best_i = max(range(len(texts)), key=lambda i: per_token[i])
        report["arms"].append({
            "diversity": arm["diversity"],
            "closed": len(texts),
            "skipped": arm["skipped"],
            "closure_rate": len(texts) / args.seeds,
            "per_token_mean": statistics.mean(per_token),
            "per_token_sd": (statistics.stdev(per_token)
                             if len(per_token) > 1 else 0.0),
            "per_token_best": max(per_token),
            "mean_letters": statistics.mean(s["letters"] for s in scores),
            "distinct_texts": len(set(texts)),
            "distinct_openings": len({opening(t) for t in texts}),
            "distinct_endings": len({ending(t) for t in texts}),
            "distinct_words": len({w for ws in words for w in ws}),
            "best_text": texts[best_i],
            "best_per_token": per_token[best_i],
        })

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)

    print(f"\n=== diversity sweep: {args.seeds} seeds per arm, "
          f"judged by {model_used} ===")
    hdr = (f"{'div':>5} {'closed':>7} {'open':>6} {'end':>5} {'vocab':>6} "
           f"{'mean/tok':>9} {'sd':>6} {'best/tok':>9} {'letters':>8}")
    print(hdr)
    for a in report["arms"]:
        if not a.get("closed"):
            print(f"{a['diversity']:>5} {'0':>7}  (no seed closed)")
            continue
        print(f"{a['diversity']:>5} {a['closed']:>3}/{args.seeds:<3} "
              f"{a['distinct_openings']:>6} {a['distinct_endings']:>5} "
              f"{a['distinct_words']:>6} {a['per_token_mean']:>9.4f} "
              f"{a['per_token_sd']:>6.4f} {a['per_token_best']:>9.4f} "
              f"{a['mean_letters']:>8.1f}")

    print("\n--- best text per arm ---")
    for a in report["arms"]:
        if a.get("closed"):
            print(f"\n[diversity={a['diversity']} "
                  f"per_token={a['best_per_token']:.4f}]")
            print(a["best_text"])
    print(f"\nreport written to {args.out}")


if __name__ == "__main__":
    main()

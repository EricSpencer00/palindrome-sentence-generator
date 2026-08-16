"""Can the coherence gain be won without saying anything? Yes — 8x over.

`docs/training.md` records that `lm_score` was raised +0.30 by a policy that
found longer words, and that the GPT-2 anchor missed it because the anchor used
the same normalization as the thing being optimized. The lesson taken from that
was to check a metric against an adversary BEFORE pointing a search at it, so
this is that check, run before any optimizer touched the number.

The gain asks whether the head's arrangement predicts the tail. Four ways to
answer yes without writing anything worth reading:

  repeat_head   the tail is the head again, verbatim and in order
  counter       "one two three four five", over and over
  one_word      a single word repeated
  real          the reference

`repeat_head` is the fatal one. It carries no information at all and the head
predicts the tail perfectly, which is exactly what the metric rewards.

    python experiments/coherence_gameability.py --n 30
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from llm_palindrome.coherence import CoherenceMetric, SelfShuffledControls, split_at_word
from llm_palindrome.lm_scoring import GPT2ConditionalScorer

from experiments.coherence_calibration import clean, load_paragraphs, truncate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--words", type=int, default=100)
    ap.add_argument("--controls", type=int, default=6)
    ap.add_argument("--skip-tokens", type=int, default=5)
    ap.add_argument("--out", default="runs/coherence_gameability.json")
    args = ap.parse_args()

    W = args.words
    paras = [clean(p) for p in load_paragraphs(args.n, W + 20, 0)]

    scorer = GPT2ConditionalScorer("gpt2", device="cpu")
    metric = CoherenceMetric(scorer, controls=["unused"], skip_tokens=args.skip_tokens)
    controls = SelfShuffledControls(n=args.controls, seed=0)

    def gain(text: str):
        head, _ = split_at_word(text)
        return metric.score(text, controls=controls(head)).gain if head.split() else None

    conditions = {
        "real": [truncate(p, W) for p in paras],
        "repeat_head": [" ".join(p.split()[:W // 2] + p.split()[:W // 2]) for p in paras],
        "one_word": ["otter " * W for _ in paras],
        "counter": [" ".join(["one two three four five"] * (W // 5)) for _ in paras],
    }

    results = {}
    for name, texts in conditions.items():
        g = [x for x in (gain(t) for t in texts) if x is not None]
        results[name] = {
            "n": len(g), "mean": round(statistics.mean(g), 4),
            "stderr": round(statistics.stdev(g) / len(g) ** 0.5, 4) if len(g) > 1 else 0.0}
        print(f"{name:14s} n={results[name]['n']:3d} "
              f"mean={results[name]['mean']:+.4f} ± {results[name]['stderr']}")

    ratio = results["repeat_head"]["mean"] / results["real"]["mean"]
    results["verdict"] = {
        "safe_as_reward": ratio < 1.0,
        "repeat_head_over_real": round(ratio, 2),
        "note": "A search maximizing this would make its tail copy its head. "
                "Usable as a diagnostic; needs a repetition guard to be a reward.",
    }
    print(f"\nrepeat_head / real = {ratio:.2f}x  →  "
          f"safe as a reward: {results['verdict']['safe_as_reward']}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"config": vars(args),
                                          "results": results}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

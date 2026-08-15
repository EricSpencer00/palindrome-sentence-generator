"""Does the palindrome get more coherent as it gets shorter?

`docs/training.md` closes on two untested ideas, and this is the cheaper one:
"coherence and length trade off directly here, and 200 letters may simply not
admit a coherent solution in a 30k vocabulary. Nothing above has tested what
the search can do at 80."

The service currently serves ~1150 letters, which is five times the length
every recorded experiment ran at, so if the tradeoff is real the deployed
configuration is at the wrong end of it.

Measured with `coherence.CoherenceMetric` under self-shuffled controls, used
here strictly as a DIAGNOSTIC. `runs/coherence_gameability.json` shows the same
number can be driven to +3.37 by making the tail repeat the head, so nothing in
this sweep may optimize against it — the search is blind to it, and each arm
only differs in its length floor.

Every run carries its own anchors, because a gain is only readable against
them: real English at the same word budget, and word salad, which is the
metric's structural zero.

    python experiments/length_sweep.py --floors 40,80,120,200,400 --seeds 24
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path

from llm_palindrome.bigram import BigramModel
from llm_palindrome.centerout import centerout_search
from llm_palindrome.coherence import CoherenceMetric, SelfShuffledControls, split_at_word
from llm_palindrome.generate import build_vocab
from llm_palindrome.lm_scoring import GPT2ConditionalScorer
from llm_palindrome.scoring import CoherentScorer
from llm_palindrome.search import WordTries
from llm_palindrome.validator import is_palindrome, normalize

from experiments.coherence_calibration import clean, load_paragraphs, truncate, word_shuffle


def summarize(name: str, values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": None, "stderr": None}
    return {"n": len(values), "mean": round(statistics.mean(values), 4),
            "stderr": round(statistics.stdev(values) / len(values) ** 0.5, 4)
            if len(values) > 1 else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--floors", default="40,80,120,200,400")
    ap.add_argument("--seeds", type=int, default=24)
    ap.add_argument("--beam", type=int, default=60)
    ap.add_argument("--budget", type=float, default=8.0)
    ap.add_argument("--diversity", type=float, default=1.0)
    ap.add_argument("--skip-tokens", type=int, default=5)
    ap.add_argument("--controls", type=int, default=6)
    ap.add_argument("--maximize", choices=["letters", "score"], default="letters",
                    help="letters: grow to the deadline, so the floor is only a "
                         "minimum. score: take the best-reading closure, which is "
                         "what actually holds length near the floor.")
    ap.add_argument("--out", default="runs/length_sweep.json")
    args = ap.parse_args()

    floors = [int(f) for f in args.floors.split(",")]

    print("loading vocabulary and bigrams...")
    vocab = build_vocab()
    tries = WordTries(vocab)
    bigrams = BigramModel.from_file("data/count_2w.txt", vocab=set(vocab))

    scorer = GPT2ConditionalScorer("gpt2", device="cpu")
    metric = CoherenceMetric(scorer, controls=["unused"], skip_tokens=args.skip_tokens)
    controls = SelfShuffledControls(n=args.controls, seed=0)

    def gain(text: str):
        head, _ = split_at_word(text)
        if not head.split():
            return None
        return metric.score(text, controls=controls(head)).gain

    results = {}
    for floor in floors:
        texts, closed = [], 0
        t0 = time.time()
        for seed in range(args.seeds):
            words = centerout_search(
                tries, CoherentScorer(bigrams), min_letters=floor,
                beam_width=args.beam, seed=seed, max_steps=10**6,
                maximize=args.maximize, candidate_limit=800,
                deadline=time.monotonic() + args.budget,
                diversity=args.diversity)
            if words and is_palindrome(" ".join(words)):
                closed += 1
                texts.append(" ".join(words))

        gains = [g for g in (gain(t) for t in texts) if g is not None]
        letters = [len(normalize(t)) for t in texts]
        results[str(floor)] = {
            "closed": f"{closed}/{args.seeds}",
            "letters_mean": round(statistics.mean(letters), 1) if letters else None,
            "words_mean": round(statistics.mean([len(t.split()) for t in texts]), 1)
            if texts else None,
            "coherence": summarize("gain", gains),
            "seconds": round(time.time() - t0, 1),
        }
        r = results[str(floor)]
        print(f"floor={floor:4d}  closed={r['closed']:6s}  letters={r['letters_mean']}"
              f"  gain={r['coherence']['mean']} ± {r['coherence']['stderr']}"
              f"  ({r['seconds']}s)")

    # Anchors, at the median word budget the arms actually produced, so the
    # comparison is not smuggling a length difference in with it.
    budget = int(statistics.median(
        [v["words_mean"] for v in results.values() if v["words_mean"]]))
    paras = [clean(p) for p in load_paragraphs(24, budget + 20, 0)]
    rng = random.Random(0)
    anchors = {
        "real_english": [truncate(p, budget) for p in paras],
        "word_salad": [truncate(word_shuffle(p, rng), budget) for p in paras],
    }
    for name, ts in anchors.items():
        g = [x for x in (gain(t) for t in ts) if x is not None]
        results[name] = {"words": budget, "coherence": summarize(name, g)}
        print(f"{name:14s} words={budget}  gain={results[name]['coherence']['mean']}"
              f" ± {results[name]['coherence']['stderr']}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"config": vars(args),
                                          "results": results}, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

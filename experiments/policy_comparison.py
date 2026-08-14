"""Do the learned scorer weights beat the hand-chosen ones?

The RL run reports its own reward, which is the thing it was optimizing and so
proves nothing. It also anchors against real GPT-2, but only on the elite
members' best candidates — a selected subset, not a fair sample.

This is the fair test: both weight vectors, every seed, same budget, every
candidate scored by GPT-2. The comparison is paired, so each seed contributes
a difference rather than two independent draws, which is what a beam search's
seed-to-seed variance demands.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from llm_palindrome.bigram import BigramModel
from llm_palindrome.generate import build_vocab
from llm_palindrome.lm_scoring import GPT2Scorer
from llm_palindrome.search import WordTries, beam_search
from llm_palindrome.textify import textify
from llm_palindrome.tunable import DEFAULT, PARAMETERS, TunableScorer
from llm_palindrome.validator import is_palindrome
from llm_palindrome.verify import verify


def run(tries, weights, bigrams, seeds, min_letters, beam):
    out = {}
    for seed in range(seeds):
        w = beam_search(tries, TunableScorer(weights, bigrams),
                        min_letters=min_letters, beam_width=beam, seed=seed)
        if w:
            out[seed] = w
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--learned", type=Path, default=Path("runs/rlvf.json"))
    ap.add_argument("--seeds", type=int, default=24)
    ap.add_argument("--min-letters", type=int, default=200)
    ap.add_argument("--beam", type=int, default=60)
    ap.add_argument("--vocab", type=int, default=30000)
    ap.add_argument("--bigrams", type=Path, default=Path("data/count_2w.txt"))
    ap.add_argument("--out", type=Path, default=Path("runs/policy_comparison.json"))
    args = ap.parse_args()

    learned = json.loads(args.learned.read_text())["learned"]
    vocab = build_vocab(args.vocab)
    tries = WordTries(vocab)
    bigrams = (BigramModel.from_file(str(args.bigrams), vocab=vocab)
               if args.bigrams.exists() else None)
    judge = GPT2Scorer("gpt2")
    vocab_set = set(vocab)

    arms = {"default": DEFAULT, "learned": learned}
    results, texts_by_arm = {}, {}
    for name, weights in arms.items():
        runs = run(tries, weights, bigrams, args.seeds, args.min_letters, args.beam)
        texts = {s: textify(w) for s, w in runs.items()}
        scores = dict(zip(texts, judge.score_texts(list(texts.values()))))
        checks = [verify(w, vocab_set) for w in runs.values()]
        texts_by_arm[name] = texts
        results[name] = {
            "weights": [round(x, 4) for x in weights],
            "closed": len(runs),
            "score_mean": round(statistics.mean(scores.values()), 4),
            "score_best": round(max(scores.values()), 4),
            "letters_mean": round(statistics.mean(c.letters for c in checks), 1),
            "short_word_rate": round(statistics.mean(c.short_word_rate for c in checks), 4),
            "distinct_ratio": round(statistics.mean(
                c.distinct_words / c.words for c in checks), 4),
            "all_valid": all(is_palindrome(t) for t in texts.values()),
            "scores": scores,
        }
        print(f"{name:8s} closed {len(runs)}/{args.seeds}  "
              f"mean {results[name]['score_mean']:+.4f}  "
              f"best {results[name]['score_best']:+.4f}  "
              f"letters {results[name]['letters_mean']:.0f}  "
              f"short {results[name]['short_word_rate']:.3f}", flush=True)

    shared = sorted(set(results["default"]["scores"]) & set(results["learned"]["scores"]))
    diffs = [results["learned"]["scores"][s] - results["default"]["scores"][s]
             for s in shared]
    if diffs:
        wins = sum(1 for d in diffs if d > 0)
        print(f"\npaired on {len(diffs)} seeds both closed: "
              f"learned - default = {statistics.mean(diffs):+.4f} "
              f"(median {statistics.median(diffs):+.4f}), "
              f"learned wins {wins}/{len(diffs)}")
        results["paired"] = {"seeds": len(diffs), "mean_delta": round(statistics.mean(diffs), 4),
                             "median_delta": round(statistics.median(diffs), 4),
                             "learned_wins": wins}

    for name in arms:
        sample = next(iter(texts_by_arm[name].values()), "")
        print(f"\n[{name}] {sample[:230]}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"parameters": PARAMETERS, **results}, indent=2))


if __name__ == "__main__":
    main()

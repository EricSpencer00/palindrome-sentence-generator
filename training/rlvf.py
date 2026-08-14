"""Learn the search's scorer weights against a mostly-verifiable reward.

The search is not differentiable and its policy is five numbers, so the method
is an evolution strategy rather than a gradient: sample weight vectors around
the current mean, run real searches, and step towards the ones that scored
well. With five parameters this is not a compromise — it is the right tool, and
it needs no backward pass through a beam search.

The reward is mostly decidable. Closure, length, repetition and word-shape come
from `verify`, which *raises* on the two properties the search guarantees
rather than scoring them, so no weight vector can trade away palindromicity or
dictionary validity to score higher. Only readability is judged, and that term
comes from the instant judge, because GPT-2 at 40ms a rollout would cap this at
a few hundred rollouts an hour.

The judge is an imitation of GPT-2, so it can be gamed. Two guards: the
readability term is bounded by the verifiable terms it cannot fake, and every
generation is re-scored with real GPT-2 on a small sample so drift between the
two shows up in the log rather than in the result.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path

from llm_palindrome.bigram import BigramModel
from llm_palindrome.generate import build_vocab
from llm_palindrome.instant_judge import InstantJudge
from llm_palindrome.search import WordTries, beam_search
from llm_palindrome.textify import textify
from llm_palindrome.tunable import DEFAULT, PARAMETERS, TunableScorer
from llm_palindrome.verify import verify


def rollout(tries, scorer, vocab_set, seed, min_letters, beam, judge,
            judge_weight, target) -> tuple[float, dict]:
    """One search, one reward. Verifiable terms first, judged term second."""
    words = beam_search(tries, scorer, min_letters=min_letters,
                        beam_width=beam, seed=seed)
    if not words:
        return -10.0, {"closed": False}

    v = verify(words, vocab_set)          # raises if the search broke a promise
    readability = judge.score(words) if judge else 0.0
    reward = v.reward(target) + judge_weight * readability
    return reward, {"closed": True, "letters": v.letters,
                    "readability": readability, "words": words}


def generation(tries, vocab_set, mean, sigma, population, rng, args, judge):
    """Sample a population of weight vectors and score each on the same seeds.

    The seeds are shared across the population on purpose: search variance
    between seeds is larger than the difference between nearby weight vectors,
    so comparing members on different seeds would rank noise.
    """
    seeds = [rng.randrange(1 << 30) for _ in range(args.rollouts)]
    members = []
    for _ in range(population):
        w = [m + sigma * rng.gauss(0, 1) for m in mean]
        rewards, closes, best = [], 0, None
        for s in seeds:
            r, info = rollout(tries, TunableScorer(w, judge.bigrams), vocab_set,
                              s, args.min_letters, args.beam, judge,
                              args.judge_weight, args.target_letters)
            rewards.append(r)
            closes += bool(info.get("closed"))
            if info.get("closed") and (best is None or r > best[0]):
                best = (r, info["words"])
        members.append({"weights": w, "reward": statistics.mean(rewards),
                        "closed": closes, "best": best})
    return members, seeds


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--judge", type=Path, default=Path("runs/instant_judge.json"))
    ap.add_argument("--bigrams", type=Path, default=Path("data/count_2w.txt"))
    ap.add_argument("--generations", type=int, default=12)
    ap.add_argument("--population", type=int, default=10)
    ap.add_argument("--elite", type=int, default=3)
    ap.add_argument("--rollouts", type=int, default=6, help="searches per member")
    ap.add_argument("--sigma", type=float, default=0.6)
    ap.add_argument("--sigma-decay", type=float, default=0.9)
    ap.add_argument("--min-letters", type=int, default=200)
    ap.add_argument("--target-letters", type=int, default=200)
    ap.add_argument("--beam", type=int, default=40)
    ap.add_argument("--judge-weight", type=float, default=4.0)
    ap.add_argument("--vocab", type=int, default=30000)
    ap.add_argument("--anchor-every", type=int, default=4,
                    help="generations between real-GPT-2 re-scoring")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("runs/rlvf.json"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    vocab = build_vocab(args.vocab)
    tries = WordTries(vocab)
    vocab_set = set(vocab)
    bigrams = (BigramModel.from_file(str(args.bigrams), vocab=vocab)
               if args.bigrams.exists() else None)
    judge = InstantJudge.load(args.judge, bigrams=bigrams)

    mean, sigma = list(DEFAULT), args.sigma
    print(f"start   {TunableScorer(mean).describe()}")

    anchor = None
    gpt2 = None  # loaded on first anchor, then kept
    history = []
    t0 = time.time()
    for gen in range(args.generations):
        members, _ = generation(tries, vocab_set, mean, sigma,
                                args.population, rng, args, judge)
        members.sort(key=lambda m: -m["reward"])
        elite = members[:args.elite]
        mean = [statistics.mean(m["weights"][i] for m in elite)
                for i in range(len(mean))]
        sigma *= args.sigma_decay

        row = {"generation": gen, "best_reward": round(elite[0]["reward"], 4),
               "mean_reward": round(statistics.mean(m["reward"] for m in members), 4),
               "closed": sum(m["closed"] for m in members),
               "weights": [round(w, 4) for w in mean], "sigma": round(sigma, 4)}

        if gen % args.anchor_every == 0 or gen == args.generations - 1:
            texts = [textify(m["best"][1]) for m in elite if m["best"]]
            if texts:
                if gpt2 is None:
                    from llm_palindrome.lm_scoring import GPT2Scorer
                    gpt2 = GPT2Scorer("gpt2")
                real = gpt2.score_texts(texts)
                row["gpt2_anchor"] = round(statistics.mean(real), 4)
                if anchor is not None:
                    row["gpt2_change"] = round(row["gpt2_anchor"] - anchor, 4)
                anchor = anchor if anchor is not None else row["gpt2_anchor"]

        history.append(row)
        print(json.dumps(row), flush=True)

    print(f"\nlearned {TunableScorer(mean).describe()}")
    print(f"default {TunableScorer(DEFAULT).describe()}")
    print(f"{time.time() - t0:.0f}s total")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"parameters": PARAMETERS, "default": DEFAULT, "learned": mean,
         "config": {k: str(v) for k, v in vars(args).items()},
         "history": history}, indent=2))


if __name__ == "__main__":
    main()

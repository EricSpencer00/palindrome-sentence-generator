"""Collect palindromes, score them with GPT-2, and fit the instant judge to it.

The training set has to look like what the judge will see, so the samples come
from the search itself rather than from a corpus: varied seeds, varied lengths,
and varied beam widths, which is what produces the spread in quality the judge
needs to learn to rank.

Held-out evaluation is by search seed, not by row. Palindromes from one seed
share words and phrasing, so splitting rows at random lets near-duplicates sit
on both sides of the split and reports a correlation the judge does not have.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path

from llm_palindrome.bigram import BigramModel
from llm_palindrome.generate import ZipfScorer, build_vocab
from llm_palindrome.instant_judge import InstantJudge
from llm_palindrome.lm_scoring import GPT2Scorer
from llm_palindrome.search import WordTries, beam_search
from llm_palindrome.textify import textify


class RandomScorer:
    """Ignores the words entirely. Produces letter-valid palindromes that read
    as badly as the dictionary allows."""

    def __init__(self, seed: int):
        self.rng = random.Random(seed)

    def word_delta(self, left, right, placement, word, growth) -> float:
        return self.rng.random()


class FrequencyOnlyScorer:
    """Common words, nothing else: no repetition penalty, no length term.
    Reads better than random and worse than the real scorer."""

    def word_delta(self, left, right, placement, word, growth) -> float:
        from wordfreq import zipf_frequency
        return zipf_frequency(word, "en")


def collect(tries, seeds, min_letters_choices, beam_choices, rng) -> list[dict]:
    """Sample across scorers as well as budgets.

    Palindromes from one scorer land within a few hundredths of each other
    under GPT-2, which is less spread than the fit needs to find a direction at
    all. Deliberately weak scorers supply the rest of the range, so the judge
    learns what bad looks like before it is asked to separate good from good.
    """
    arms = [("zipf", lambda s: ZipfScorer()),
            ("freq_only", lambda s: FrequencyOnlyScorer()),
            ("random", lambda s: RandomScorer(s))]
    out = []
    for seed in range(seeds):
        name, make = arms[seed % len(arms)]
        min_letters = rng.choice(min_letters_choices)
        beam = rng.choice(beam_choices)
        t0 = time.time()
        words = beam_search(tries, make(seed), min_letters=min_letters,
                            beam_width=beam, seed=seed)
        if not words:
            continue
        out.append({"seed": seed, "arm": name, "words": words,
                    "min_letters": min_letters, "beam": beam,
                    "seconds": round(time.time() - t0, 2)})
        if len(out) % 50 == 0:
            print(f"  collected {len(out)}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=400)
    ap.add_argument("--judge-model", default="gpt2")
    ap.add_argument("--vocab", type=int, default=30000)
    ap.add_argument("--bigrams", type=Path, default=Path("data/count_2w.txt"))
    ap.add_argument("--samples-out", type=Path, default=Path("runs/judge_samples.json"))
    ap.add_argument("--out", type=Path, default=Path("runs/instant_judge.json"))
    ap.add_argument("--reuse-samples", action="store_true",
                    help="skip collection and refit from --samples-out")
    ap.add_argument("--target", choices=["per_letter", "per_token"],
                    default="per_token",
                    help="which normalization of GPT-2's score to imitate")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    vocab = build_vocab(args.vocab)
    bigrams = (BigramModel.from_file(str(args.bigrams), vocab=vocab)
               if args.bigrams.exists() else None)
    if bigrams is None:
        print(f"no bigram file at {args.bigrams}; bigram features will be flat")

    if args.reuse_samples and args.samples_out.exists():
        rows = json.loads(args.samples_out.read_text())
        print(f"reusing {len(rows)} samples")
    else:
        tries = WordTries(vocab)
        print(f"collecting from {args.seeds} seeds...", flush=True)
        rows = collect(tries, args.seeds, [120, 200, 300], [30, 60, 90], rng)
        print(f"scoring {len(rows)} palindromes with {args.judge_model}...", flush=True)
        judge = GPT2Scorer(args.judge_model)
        scores = judge.score_texts([textify(r["words"]) for r in rows])
        for r, s in zip(rows, scores):
            r["gpt2"] = s
        args.samples_out.write_text(json.dumps(rows, indent=2))

    # Imitating the per-letter score would hand the RL loop the same exploit it
    # already found there: longer words cost fewer tokens per letter, so that
    # number rises without the text reading better. Converting costs nothing —
    # total logprob is the stored score times the letter count, and dividing by
    # the token count instead needs only a tokenizer, not another GPT-2 pass.
    if args.target == "per_token":
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.judge_model)
        for r in rows:
            text = textify(r["words"])
            letters = max(1, sum(c.isalpha() for c in text))
            n_tok = max(1, len(tok(text)["input_ids"]))
            r["gpt2_per_letter"] = r["gpt2"]
            r["gpt2"] = r["gpt2"] * letters / n_tok
        print(f"target: GPT-2 score per token "
              f"(converted from per letter, no rescoring)")

    # Split by seed so near-duplicate palindromes cannot straddle the split.
    seeds = sorted({r["seed"] for r in rows})
    rng.shuffle(seeds)
    cut = int(0.75 * len(seeds))
    train_seeds = set(seeds[:cut])
    train = [(r["words"], r["gpt2"]) for r in rows if r["seed"] in train_seeds]
    test_rows = [r for r in rows if r["seed"] not in train_seeds]
    test = [(r["words"], r["gpt2"]) for r in test_rows]
    print(f"train {len(train)}  test {len(test)}  "
          f"target sd {statistics.pstdev([y for _, y in train]):.3f}")

    model = InstantJudge(bigrams=bigrams).fit(train)
    print("\nweights:")
    print(model.explain())
    print("\ntrain:", json.dumps(model.evaluate(train)))
    print("test :", json.dumps(model.evaluate(test)))

    # The honest test. Telling good palindromes from random ones is easy and
    # not what the judge is for; ranking within the arm the search actually
    # uses is the job, and it is a much narrower target.
    hard = [(r["words"], r["gpt2"]) for r in test_rows if r.get("arm") == "zipf"]
    if len(hard) >= 3:
        print("test, zipf arm only:", json.dumps(model.evaluate(hard)))

    t0 = time.perf_counter()
    for words, _ in test:
        model.score(words)
    per = (time.perf_counter() - t0) / max(1, len(test))
    print(f"\nspeed: {per * 1e6:.0f} us per palindrome")

    model.save(args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()

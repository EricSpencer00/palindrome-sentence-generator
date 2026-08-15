"""Does the coherence metric measure coherence? Calibrate it before trusting it.

`CoherenceMetric` claims to detect whether a text's own opening informs its own
ending. That claim is testable without any palindromes at all, and it has to be
tested first, because a metric that cannot separate real prose from prose with
its sentences shuffled would happily certify anything downstream.

Five conditions, all truncated to the same word budget so length is not a
confound (the metric averages over the tail, and a longer tail is a different
measurement):

  real          wikitext paragraph, untouched
  sent_shuffled its sentences reordered — LOCALLY fluent, globally scrambled
  word_shuffled every word reordered — the floor
  stitched      first half of paragraph A, second half of paragraph B
  palindrome    what this project currently generates

`stitched` is the one that defines zero. Its tail's own head is, by
construction, foreign to it — exactly the condition the control prefixes create
— so a correct metric must score it at approximately 0.0. If it does not, the
gain is picking up something other than topic and the number is not usable.

`sent_shuffled` is the condition that matters for this project: it is what
"locally fluent, globally incoherent" looks like in real English, and it is the
bar the palindrome output has to clear for anything to have been achieved.

    python experiments/coherence_calibration.py --n 40 --words 100
"""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import time
from pathlib import Path

from llm_palindrome.coherence import CoherenceMetric, SelfShuffledControls, split_at_word
from llm_palindrome.lm_scoring import GPT2ConditionalScorer


def load_paragraphs(n: int, min_words: int, seed: int) -> list[str]:
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    paras = [t.strip() for t in ds["text"]
             if len(t.split()) >= min_words and not t.strip().startswith("=")]
    random.Random(seed).shuffle(paras)
    return paras[:n]


def clean(text: str) -> str:
    """Strip wikitext's tokenizer artifacts so the model sees ordinary prose."""
    text = text.replace("@-@", "-").replace("@,@", ",").replace("@.@", ".")
    return re.sub(r"\s+", " ", text).strip()


def truncate(text: str, words: int) -> str:
    return " ".join(text.split()[:words])


def sentence_shuffle(text: str, rng: random.Random) -> str:
    sents = [s.strip() for s in re.split(r"(?<=[.!?]) +", text) if s.strip()]
    rng.shuffle(sents)
    return " ".join(sents)


def word_shuffle(text: str, rng: random.Random) -> str:
    ws = text.split()
    rng.shuffle(ws)
    return " ".join(ws)


def generate_palindromes(n: int, min_letters: int, budget: float) -> list[str]:
    """Fresh output from the same search the service runs."""
    from llm_palindrome.bigram import BigramModel
    from llm_palindrome.centerout import centerout_search
    from llm_palindrome.generate import build_vocab
    from llm_palindrome.scoring import CoherentScorer
    from llm_palindrome.search import WordTries

    vocab = build_vocab()
    tries = WordTries(vocab)
    bigrams = BigramModel.from_file("data/count_2w.txt", vocab=set(vocab))

    out = []
    for seed in range(n):
        words = centerout_search(
            tries, CoherentScorer(bigrams), min_letters=min_letters,
            beam_width=60, seed=seed, max_steps=10**6, maximize="letters",
            candidate_limit=800, deadline=time.monotonic() + budget,
            # The service pins seed=0 and gets one text; the sweep needs
            # genuinely different ones, and `docs/training.md` shows 0.4 is too
            # narrow for the jitter to reorder anything.
            diversity=1.0,
        )
        if words:
            out.append(" ".join(words))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--words", type=int, default=100, help="word budget per text")
    ap.add_argument("--controls", type=int, default=6)
    ap.add_argument("--skip-tokens", type=int, default=5)
    ap.add_argument("--min-letters", type=int, default=400)
    ap.add_argument("--budget", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--control-mode", choices=["foreign", "self_shuffled"],
                    default="self_shuffled",
                    help="foreign: prefixes from other texts (measures vocabulary reuse, falsified). self_shuffled: the text's own head reordered (measures arrangement).")
    ap.add_argument("--out", default="runs/coherence_calibration.json")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    paras = [clean(p) for p in load_paragraphs(args.n * 2 + args.controls,
                                               args.words + 20, args.seed)]

    # Control prefixes are held out: never one of the texts being scored.
    controls = [truncate(p, args.words // 2) for p in paras[:args.controls]]
    body = paras[args.controls:]
    subjects, donors = body[:args.n], body[args.n:args.n * 2]

    print(f"generating {args.n} palindromes at >={args.min_letters} letters...")
    pals = generate_palindromes(args.n, args.min_letters, args.budget)
    print(f"  {len(pals)} closed")

    conditions: dict[str, list[str]] = {
        "real": [truncate(p, args.words) for p in subjects],
        "sent_shuffled": [truncate(sentence_shuffle(p, rng), args.words) for p in subjects],
        "word_shuffled": [truncate(word_shuffle(p, rng), args.words) for p in subjects],
        "stitched": [truncate(" ".join(a.split()[:args.words // 2]
                                       + b.split()[:args.words // 2]), args.words)
                     for a, b in zip(subjects, donors)],
        "palindrome": [truncate(p, args.words) for p in pals],
    }

    scorer = GPT2ConditionalScorer("gpt2", device="cpu")
    metric = CoherenceMetric(scorer, controls=controls, skip_tokens=args.skip_tokens)
    self_controls = SelfShuffledControls(n=args.controls, seed=args.seed)

    def gain_of(text: str):
        if args.control_mode == "foreign":
            return metric.score(text).gain
        head, _ = split_at_word(text)
        return metric.score(text, controls=self_controls(head)).gain

    results = {}
    for name, texts in conditions.items():
        t0 = time.time()
        gains = [g for g in (gain_of(t) for t in texts) if g is not None]
        results[name] = {
            "n": len(gains),
            "mean": round(statistics.mean(gains), 4),
            "sd": round(statistics.stdev(gains), 4) if len(gains) > 1 else None,
            "stderr": round(statistics.stdev(gains) / len(gains) ** 0.5, 4)
            if len(gains) > 1 else None,
            "median": round(statistics.median(gains), 4),
            "gains": [round(g, 4) for g in gains],
        }
        print(f"{name:16s} n={len(gains):3d} mean={results[name]['mean']:+.4f} "
              f"± {results[name]['stderr']}  ({time.time() - t0:.0f}s)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"config": vars(args), "results": results,
         "samples": {k: v[:2] for k, v in conditions.items()}}, indent=2))
    print(f"\nwrote {args.out}")

    # real vs sent_shuffled is a PAIRED comparison — the same paragraphs, one
    # transformed — so the unpaired stderr above understates the power badly.
    pairs = list(zip(results["real"]["gains"], results["sent_shuffled"]["gains"]))
    diffs = [a - b for a, b in pairs]
    d_mean = statistics.mean(diffs)
    d_se = statistics.stdev(diffs) / len(diffs) ** 0.5
    wins = sum(1 for d in diffs if d > 0)
    results["paired_real_vs_sent_shuffled"] = {
        "mean_diff": round(d_mean, 4), "stderr": round(d_se, 4),
        "t": round(d_mean / d_se, 2), "wins": wins, "n": len(diffs)}
    print(f"\npaired real - sent_shuffled: {d_mean:+.4f} ± {d_se:.4f} "
          f"(t={d_mean / d_se:.2f}, wins {wins}/{len(diffs)})")

    real = results["real"]["mean"]
    sent = results["sent_shuffled"]["mean"]
    word = results["word_shuffled"]["mean"]
    print("\nvalidity — the checks the first design failed:")
    print(f"  real beats sentence-shuffled (paired t>2): "
          f"{d_mean / d_se > 2}  ({real:+.4f} vs {sent:+.4f}, t={d_mean / d_se:.2f})")
    print(f"  word-shuffled does NOT beat real:          "
          f"{word < real}  ({word:+.4f} vs {real:+.4f})")
    print(f"  word-shuffled near its structural zero:    "
          f"{abs(word) < 0.1}  ({word:+.4f})")


if __name__ == "__main__":
    main()

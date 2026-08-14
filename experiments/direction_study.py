"""Does growth direction matter? Outside-in vs center-out at matched budgets.

Two hypotheses:

H1 (closure) Outside-in closes more often, because it finishes when its slack
   is merely PALINDROMIC while center-out needs slack exactly EMPTY.

H2 (fluency profile) Each method commits to good text where it starts and
   pushes its compromises to where it finishes. Outside-in starts at the two
   ends, so its opening and closing should read better than its middle.
   Center-out starts in the middle, so the profile should invert. Readers
   weight openings and endings most, which would favor outside-in.
"""
from __future__ import annotations

import json
import statistics
import time

from llm_palindrome.centerout import centerout_search
from llm_palindrome.generate import ZipfScorer, build_vocab
from llm_palindrome.lm_scoring import GPT2Scorer
from llm_palindrome.search import WordTries, beam_search
from llm_palindrome.textify import textify
from llm_palindrome.validator import is_palindrome, normalize

CENTERS = ["", "a", "o", "i", "level", "madam", "noon", "deed", "racecar", "ere"]
SEEDS = 24
MIN_LETTERS = 200
BEAM = 60


def thirds(text: str, lm) -> tuple[float, float, float]:
    """Mean LM score of the first, middle, and last third of the sentences."""
    sents = [s.strip() + "." for s in text.split(".") if s.strip()]
    if len(sents) < 3:
        return (float("nan"),) * 3
    scores = lm.score_texts(sents)
    n = len(scores)
    cut = max(1, n // 3)
    head, mid, tail = scores[:cut], scores[cut:n - cut], scores[n - cut:]
    return (statistics.mean(head),
            statistics.mean(mid) if mid else float("nan"),
            statistics.mean(tail))


def main() -> None:
    tries = WordTries(build_vocab())
    scorer = ZipfScorer()
    lm = GPT2Scorer("gpt2")
    results = {}

    for arm in ("outside_in", "center_out"):
        t0 = time.time()
        texts = []
        for seed in range(SEEDS):
            if arm == "outside_in":
                words = beam_search(tries, scorer, min_letters=MIN_LETTERS,
                                    beam_width=BEAM, seed=seed)
            else:
                words = centerout_search(tries, scorer, min_letters=MIN_LETTERS,
                                         beam_width=BEAM, seed=seed,
                                         center=CENTERS[seed % len(CENTERS)])
            if words:
                texts.append(textify(words))
        elapsed = time.time() - t0

        if not texts:
            results[arm] = {"closure_rate": 0.0, "seconds": round(elapsed, 1)}
            print(f"{arm}: closed 0/{SEEDS} in {elapsed:.1f}s")
            continue

        overall = lm.score_texts(texts)
        best_i = max(range(len(texts)), key=lambda i: overall[i])
        profiles = [thirds(t, lm) for t in texts]
        valid = [p for p in profiles if p[0] == p[0] and p[1] == p[1]]

        results[arm] = {
            "closure_rate": round(len(texts) / SEEDS, 3),
            "seconds": round(elapsed, 1),
            "mean_letters": round(statistics.mean(len(normalize(t)) for t in texts), 1),
            "best_lm": round(overall[best_i], 3),
            "mean_lm": round(statistics.mean(overall), 3),
            "all_valid": all(is_palindrome(t) for t in texts),
            "head_third": round(statistics.mean(p[0] for p in valid), 3),
            "middle_third": round(statistics.mean(p[1] for p in valid), 3),
            "tail_third": round(statistics.mean(p[2] for p in valid), 3),
            "best_text": texts[best_i],
        }
        r = results[arm]
        print(f"\n=== {arm} ===")
        print(f"closed {len(texts)}/{SEEDS}  {elapsed:.1f}s  "
              f"mean_letters={r['mean_letters']}  all_valid={r['all_valid']}")
        print(f"lm best={r['best_lm']}  mean={r['mean_lm']}")
        print(f"fluency by position:  head={r['head_third']}  "
              f"middle={r['middle_third']}  tail={r['tail_third']}")
        print(f"  {r['best_text'][:110]}...")

    with open("experiments/direction_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nwrote experiments/direction_results.json")


if __name__ == "__main__":
    main()

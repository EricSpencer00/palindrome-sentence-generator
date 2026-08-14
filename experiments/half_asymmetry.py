"""Which half of a palindrome reads better, and why?

Both growth directions build one half by APPENDING words (left-to-right, the
direction English is written) and the other by PREPENDING (right-to-left,
against the grain). The halves are forced to share a letter sequence, so any
fluency gap between them isolates the cost of building text backwards.

  outside-in  appends on the LEFT  (left half natural)
  center-out  appends on the RIGHT (right half natural)

If the natural-order half wins in BOTH arms, the gap is caused by construction
direction rather than by position in the text.
"""
from __future__ import annotations

import statistics

from llm_palindrome.centerout import centerout_search
from llm_palindrome.generate import ZipfScorer, build_vocab
from llm_palindrome.lm_scoring import GPT2Scorer
from llm_palindrome.search import WordTries, beam_search
from llm_palindrome.textify import textify

SEEDS = 24
CENTERS = ["", "a", "o", "i", "level", "madam", "noon", "deed", "racecar", "ere"]


def split_halves(words: list[str]) -> tuple[str, str]:
    """Split the word sequence at the midpoint of its letters."""
    total = sum(len(w) for w in words)
    acc, cut = 0, len(words)
    for i, w in enumerate(words):
        acc += len(w)
        if acc >= total / 2:
            cut = i + 1
            break
    return " ".join(words[:cut]), " ".join(words[cut:])


def main() -> None:
    tries = WordTries(build_vocab())
    scorer = ZipfScorer()
    lm = GPT2Scorer("gpt2")

    for arm in ("outside_in", "center_out"):
        seqs = []
        for seed in range(SEEDS):
            if arm == "outside_in":
                w = beam_search(tries, scorer, min_letters=200,
                                beam_width=60, seed=seed)
            else:
                w = centerout_search(tries, scorer, min_letters=200,
                                     beam_width=60, seed=seed,
                                     center=CENTERS[seed % len(CENTERS)])
            if w:
                seqs.append(w)

        lefts, rights = zip(*(split_halves(s) for s in seqs))
        ls = lm.score_texts([textify(t.split()) for t in lefts])
        rs = lm.score_texts([textify(t.split()) for t in rights])
        natural = "left" if arm == "outside_in" else "right"
        nat = ls if natural == "left" else rs
        rev = rs if natural == "left" else ls

        print(f"\n=== {arm} (natural-order half = {natural}) ===")
        print(f"  left  half: {statistics.mean(ls):+.3f}")
        print(f"  right half: {statistics.mean(rs):+.3f}")
        print(f"  natural-order half beats reversed half by "
              f"{statistics.mean(nat) - statistics.mean(rev):+.3f}")
        wins = sum(1 for a, b in zip(nat, rev) if a > b)
        print(f"  natural half wins in {wins}/{len(nat)} candidates")


if __name__ == "__main__":
    main()

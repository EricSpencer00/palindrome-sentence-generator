"""Spend the punctuation, which the mirror cannot see.

`normalize` strips case and punctuation, so a palindrome's letters are the only
thing the constraint constrains. Where the periods, commas and question marks
fall is a free variable, and this project has been spending it on one heuristic
--- `textify.segment_at_weak_joins` cuts at the joins a bigram model likes
least, which puts a break where the text has already fallen apart.

That is a reasonable default and it is not a search. This script asks the
question properly: over all ways of cutting a word sequence into runs, which
one exposes the most syntactic structure? Punctuation is allowed to be
rudimentary and even wrong in the way spoken English is wrong --- "like, the
dog. maybe has? something to do with this?" --- because a fragment that is
punctuated as a fragment reads better than the same fragment inside a sentence
that never arrives.

Tiers
-----
Each candidate run is scored by the strongest test it passes, using the Brown
tag tables in `llm_palindrome/syntax.py`:

    SENTENCE  `sentence_like`: an attested whole-sentence tag shape, with a
              verb and a subject-shaped opening, all in the SAME reading
    SHAPED    `shaped`: some reading is a tag shape a whole Brown sentence has,
              but without the subject-and-verb requirement
    PHRASE    `plausible`: every tag trigram of some reading occurs in Brown;
              locally well-formed and nothing more
    NONE      no reading survives, or a word Brown never tagged

and each tier gets the mark that suits how much it is claiming:

    SENTENCE  .     it is a sentence
    SHAPED    ?     it has the shape of one; the question mark carries the
                    intonation that makes an unanchored clause readable
    PHRASE    ,     it holds together and does not stand alone
    NONE      --    a break, admitting that nothing here parses

Maximising coverage
-------------------
A dynamic programme over cut positions maximises the number of words sitting
inside a run of tier SHAPED or better, with SENTENCE weighted above SHAPED.
The DP is total, because a run of tier NONE is always allowed, so every word
sequence gets punctuated and the score says how much of it survived.

Controls
--------
The number is meaningless alone. Every run reports three arms over the same
word counts:

    prose    real English sentences with their own punctuation removed
    salad    those same prose words, shuffled
    target   the text under test
    selfshuf THE TARGET'S OWN WORDS, shuffled

The fourth arm is the one that decides anything, and the first three nearly
misled us without it. A palindrome from a frequency-ranked vocabulary is made
of short words, and short words carry more Brown tags: 2.18 per word against
prose's 1.80, at mean word lengths of 2.79 and 4.97. More tags means more tag
readings per run, which means more chances that some reading matches some
attested shape. Against prose the target therefore scores well for a reason
that has nothing to do with its structure --- the same word-length confound
that gamed the per-letter LM score in `docs/training.md`.

Shuffling the target's own words holds vocabulary, length and ambiguity fixed
and destroys only the order. The gap between `target` and `selfshuf` is what
the word order bought, and it is the only number here that is about
structure.

Usage
-----
    python experiments/punctuation_search.py                  # repo fallbacks
    python experiments/punctuation_search.py --words-json run.json
    python experiments/punctuation_search.py --show 2
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
from typing import NamedTuple, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_palindrome.syntax import brown_tables, plausible, shaped, sentence_like
from llm_palindrome.validator import normalize

# Tier order matters: index is the tier, and higher is stronger.
NONE, PHRASE, SHAPED, SENTENCE = 0, 1, 2, 3
MARK = {SENTENCE: ".", SHAPED: "?", PHRASE: ",", NONE: " --"}
TIER_NAME = {SENTENCE: "sentence", SHAPED: "shaped", PHRASE: "phrase", NONE: "none"}

# Weight per word, by the tier of the run it sits in. SENTENCE is worth more
# than SHAPED per word so the DP prefers a real clause to a longer shape, and
# PHRASE is worth little so it cannot outbid a shorter SHAPED run.
WEIGHT = {SENTENCE: 3.0, SHAPED: 2.0, PHRASE: 0.4, NONE: 0.0}

MIN_RUN, MAX_RUN = 3, 9


class Segmentation(NamedTuple):
    runs: list[tuple[list[str], int]]      # (words, tier)
    score: float
    covered: float                          # fraction of words at SHAPED+
    sentences: float                        # fraction of words at SENTENCE


def tier_of(words: Sequence[str], table, shapes, trigrams) -> int:
    """The strongest test this run passes."""
    if sentence_like(words, table, shapes):
        return SENTENCE
    if shaped(words, table, shapes):
        return SHAPED
    if plausible(words, table, trigrams):
        return PHRASE
    return NONE


def punctuate(words: Sequence[str], table, shapes, trigrams) -> Segmentation:
    """Cut `words` into runs so that as much as possible sits in a good one.

    best[i] is the score of the best segmentation of words[:i]. A run of tier
    NONE is always available at any length, so best[i] is never unreachable and
    no word sequence can fail to be punctuated.
    """
    n = len(words)
    cache: dict[tuple[int, int], int] = {}

    def run_tier(i: int, j: int) -> int:
        key = (i, j)
        if key not in cache:
            cache[key] = tier_of(words[i:j], table, shapes, trigrams)
        return cache[key]

    best = [float("-inf")] * (n + 1)
    back: list[tuple[int, int]] = [(-1, NONE)] * (n + 1)
    best[0] = 0.0
    for j in range(1, n + 1):
        for i in range(max(0, j - MAX_RUN), j):
            if best[i] == float("-inf"):
                continue
            length = j - i
            tier = run_tier(i, j) if length >= MIN_RUN else NONE
            gain = WEIGHT[tier] * length
            if best[i] + gain > best[j]:
                best[j] = best[i] + gain
                back[j] = (i, tier)

    runs: list[tuple[list[str], int]] = []
    j = n
    while j > 0:
        i, tier = back[j]
        runs.append((list(words[i:j]), tier))
        j = i
    runs.reverse()

    covered = sum(len(w) for w, t in runs if t >= SHAPED) / max(1, n)
    sent = sum(len(w) for w, t in runs if t == SENTENCE) / max(1, n)
    return Segmentation(runs, best[n], covered, sent)


def render(seg: Segmentation) -> str:
    out = []
    for words, tier in seg.runs:
        text = " ".join(words)
        out.append(text[0].upper() + text[1:] + MARK[tier])
    return " ".join(out)


# ------------------------------------------------------------------- corpora

def prose_words(n_words: int, rng: random.Random) -> list[str]:
    """Real English, punctuation removed, so the procedure sees what we see."""
    import glob
    import re
    pattern = os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--wikitext/snapshots/*/"
        "wikitext-2-raw-v1/train-*.parquet")
    hits = glob.glob(pattern)
    if not hits:
        raise SystemExit("no corpus found: expected a cached wikitext-2 parquet")
    import pyarrow.parquet as pq
    lines = [t.strip() for t in pq.read_table(hits[0], columns=["text"])
             .column("text").to_pylist()
             if t.strip() and not t.strip().startswith("=")]
    rng.shuffle(lines)
    words: list[str] = []
    for line in lines:
        for w in line.split():
            w = normalize(w)
            if w:
                words.append(w)
        if len(words) >= n_words:
            break
    return words[:n_words]


def load_targets(path: str | None) -> list[list[str]]:
    path = path or "data/fallback_texts.json"
    data = json.load(open(path))
    out = []
    for entry in (data if isinstance(data, list) else [data]):
        if isinstance(entry, dict):
            words = entry.get("words") or (entry.get("left", []) + entry.get("right", []))
        else:
            words = str(entry).split()
        words = [normalize(w) for w in words]
        if any(words):
            out.append([w for w in words if w])
    return out


# ---------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--words-json", default=None,
                    help="file with {'words': [...]} or {'left','right'} entries")
    ap.add_argument("--show", type=int, default=1, help="print N punctuated texts")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="experiments/punctuation_search.json")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    print("loading Brown tag tables ...", flush=True)
    table, shapes, trigrams = brown_tables()
    print(f"  {len(table)} tagged words, {len(shapes)} sentence shapes\n")

    targets = load_targets(args.words_json)
    if not targets:
        raise SystemExit("no target texts")

    rows = []
    for idx, words in enumerate(targets):
        n = len(words)
        arms = {
            "target": words,
            "prose": prose_words(n, random.Random(args.seed + idx)),
        }
        salad = list(arms["prose"])
        random.Random(args.seed + idx).shuffle(salad)
        arms["salad"] = salad
        # The control that holds vocabulary fixed and destroys only the order.
        selfshuf = list(words)
        random.Random(args.seed + 1000 + idx).shuffle(selfshuf)
        arms["selfshuf"] = selfshuf

        row = {"index": idx, "n_words": n}
        for name, seq in arms.items():
            seg = punctuate(seq, table, shapes, trigrams)
            row[name] = {"covered": seg.covered, "sentences": seg.sentences,
                         "score_per_word": seg.score / max(1, n)}
            if name == "target" and idx < args.show:
                print(f"--- text {idx}, {n} words ---")
                print(render(seg))
                print()
        rows.append(row)
        gap = row["target"]["sentences"] - row["selfshuf"]["sentences"]
        print(f"text {idx:2d} n={n:4d}  sentence-tier: "
              f"target {row['target']['sentences']:.3f}  "
              f"selfshuf {row['selfshuf']['sentences']:.3f}  "
              f"(order buys {gap:+.3f})   "
              f"prose {row['prose']['sentences']:.3f}  "
              f"salad {row['salad']['sentences']:.3f}")

    with open(args.out, "w") as fh:
        json.dump(rows, fh, indent=2)

    print()
    for arm in ("target", "selfshuf", "prose", "salad"):
        cov = statistics.fmean(r[arm]["covered"] for r in rows)
        sen = statistics.fmean(r[arm]["sentences"] for r in rows)
        print(f"{arm:9s} covered {cov:.3f}   sentence-tier {sen:.3f}")
    gaps = [r["target"]["sentences"] - r["selfshuf"]["sentences"] for r in rows]
    mean = statistics.fmean(gaps)
    sd = statistics.stdev(gaps) if len(gaps) > 1 else 0.0
    wins = sum(g > 0 for g in gaps)
    print(f"\nwhat the word ORDER buys: {mean:+.3f} sentence-tier coverage "
          f"(sd {sd:.3f}), target beats its own shuffle in {wins}/{len(gaps)}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

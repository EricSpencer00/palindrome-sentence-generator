"""What the mirror costs, in bits per free letter.

The claim this script exists to make reproducible: writing English under a
palindrome constraint is more expensive than writing English, by an amount that
can be stated in bits per letter and does not depend much on how long the span
is, how it is segmented, or which model does the scoring.

The measurement
---------------
A palindrome of 2k letters has k free letters. Each free letter is placed
twice: once in the reading that runs left to right, once in the reading that
runs right to left. Both readings have to be English.

So take real English, strip it to letters, and score it two ways under one
model and one vocabulary:

    forward   the letters in their own order, re-segmented into words
    reversed  the same letters in the opposite order, re-segmented into words

The forward number is what English costs. The reversed number is what the same
letters cost when they are asked to be English in the other direction as well.
The difference is the price of the mirror, per free letter.

Why both sides are re-segmented
-------------------------------
The obvious comparison — original text with its own spacing, against reversed
text re-segmented — confounds two things. Optimal re-segmentation under a
unigram model is not free; it puts boundaries where a word model likes them,
not where the writer did. Running the identical procedure on both directions
removes that, and what is left is directional. The natural-spacing number is
reported too, as the difference between the two says what re-segmentation
alone costs.

Why segmentation is allowed to fail into single letters
-------------------------------------------------------
Reversed English almost never segments cleanly into a dictionary: at twenty
letters it happens in none of a first sample of thirty, so an estimator that
drops the spans which do not segment has nothing left to average. Both
directions are therefore segmented over the vocabulary PLUS the twenty-six
bare letters, which makes segmentation always succeed and hands the penalty to
the language model instead of to a constant chosen here. The forward direction
gets the same escape and hardly uses it, which is the point.

Dictionary coverage — the fraction of letters that land inside a real word of
three letters or more — is reported alongside, because it says how much of the
gap is words the model dislikes and how much is not words at all.

Usage
-----
    python experiments/mirror_cost.py                       # gpt2, all sweeps
    python experiments/mirror_cost.py --models gpt2 gpt2-large Qwen/Qwen2.5-0.5B
    python experiments/mirror_cost.py --spans 20 40 80 --n 200
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import re
import statistics
import sys
from functools import lru_cache
from typing import NamedTuple, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_palindrome.shortwords import is_real_short
from llm_palindrome.validator import normalize

NATS_PER_BIT = math.log(2.0)

# Per-word cost in the unigram segmentation objective, in log10 units to match
# wordfreq's Zipf scale. Without it a run of "a"s tiles anything for free.
# Inherited from llm_palindrome/respace.py so the two agree.
WORD_COST = 9.0
MAX_WORD = 20


# ---------------------------------------------------------------- vocabulary

def load_vocab(path: str) -> frozenset[str]:
    """The dictionary both readings are segmented into.

    One vocabulary for both directions, so nothing in the comparison can come
    from one side having more words available than the other.
    """
    with open(path) as fh:
        words = [w.strip().lower() for w in fh if w.strip()]
    return frozenset(w for w in words
                     if w.isalpha() and w.isascii() and is_real_short(w))


@lru_cache(maxsize=None)
def _zipf(word: str) -> float:
    from wordfreq import zipf_frequency
    return zipf_frequency(word, "en")


# -------------------------------------------------------------- segmentation

ALPHABET = frozenset("abcdefghijklmnopqrstuvwxyz")


def segment(letters: str, vocab: frozenset[str], strategy: str) -> list[str]:
    """Split `letters` into words, falling back to bare letters where needed.

    Three objectives, so the result can be checked for dependence on any one
    of them:

    unigram   maximise summed Zipf frequency less WORD_COST per word
    fewest    minimise the number of units (longest words, frequency ignored)
    greedy    longest match left to right, with backtracking

    Under `unigram` a non-word letter scores zipf 0 and still pays WORD_COST,
    so it is always the last resort; under the other two it costs the same as
    any other unit and longest-match keeps it rare. Segmentation never fails,
    so no span is silently dropped from an average.
    """
    n = len(letters)
    if n == 0:
        return []

    if strategy == "greedy":
        return _greedy(letters, vocab, 0, {})

    # Dynamic programme over letter positions. best[i] is the score of the
    # best segmentation of letters[:i]; back[i] is where its last word starts.
    neg_inf = float("-inf")
    best = [neg_inf] * (n + 1)
    back = [-1] * (n + 1)
    best[0] = 0.0
    for i in range(1, n + 1):
        for j in range(max(0, i - MAX_WORD), i):
            if best[j] == neg_inf:
                continue
            word = letters[j:i]
            if word not in vocab and not (len(word) == 1 and word in ALPHABET):
                continue
            if strategy == "unigram":
                gain = (_zipf(word) if word in vocab else 0.0) - WORD_COST
            elif strategy == "fewest":
                gain = -1.0
            else:
                raise ValueError(f"unknown strategy {strategy!r}")
            if best[j] + gain > best[i]:
                best[i] = best[j] + gain
                back[i] = j
    assert best[n] != neg_inf, "single-letter fallback should make this total"

    words: list[str] = []
    i = n
    while i > 0:
        j = back[i]
        words.append(letters[j:i])
        i = j
    words.reverse()
    return words


def _greedy(letters: str, vocab: frozenset[str], i: int,
            memo: dict[int, list[str]]) -> list[str]:
    if i == len(letters):
        return []
    if i in memo:
        return memo[i]
    for end in range(min(len(letters), i + MAX_WORD), i, -1):
        word = letters[i:end]
        if word not in vocab and end != i + 1:
            continue
        memo[i] = [word] + _greedy(letters, vocab, end, memo)
        return memo[i]
    raise AssertionError("unreachable: a single letter always matches")


def coverage(words: Sequence[str], vocab: frozenset[str]) -> float:
    """Fraction of letters sitting inside a real word of three or more.

    One- and two-letter units are excluded even when they are words, because
    that is the corner where a segmenter can tile anything and claim success
    (llm_palindrome/shortwords.py makes the same judgement for the search).
    """
    total = sum(len(w) for w in words)
    if not total:
        return 0.0
    real = sum(len(w) for w in words if len(w) >= 3 and w in vocab)
    return real / total


# ------------------------------------------------------------------- scoring

class Scorer:
    """Total token log-probability of a text, under one causal LM.

    Returns nats. The caller divides by letters, never by tokens: this project
    has been burned by per-token normalisation before (docs/training.md), and
    here the letter is the unit the constraint is denominated in. Both
    directions are scored over the SAME letters, so the denominator is
    identical on both sides and cannot move the difference.
    """

    def __init__(self, model_name: str, device: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.name = model_name
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.model.eval()

    def total_logprob(self, text: str) -> float | None:
        torch = self.torch
        ids = self.tok(text, add_special_tokens=False).input_ids
        if len(ids) < 2:
            return None
        with torch.no_grad():
            tensor = torch.tensor([ids], device=self.device)
            logits = self.model(tensor).logits
            logprobs = torch.log_softmax(logits[0, :-1].float(), dim=-1)
            picked = logprobs.gather(-1, tensor[0, 1:].unsqueeze(-1)).squeeze(-1)
        # The first token is unscored under every model, on both sides alike.
        return float(picked.sum())


# -------------------------------------------------------------------- corpus

def load_corpus(limit_chars: int = 4_000_000) -> str:
    """Plain English prose. wikitext-2 if it is cached, else the repo's own."""
    pattern = os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--wikitext/snapshots/*/"
        "wikitext-2-raw-v1/train-*.parquet")
    hits = glob.glob(pattern)
    if hits:
        import pyarrow.parquet as pq
        table = pq.read_table(hits[0], columns=["text"])
        chunks: list[str] = []
        total = 0
        for value in table.column("text").to_pylist():
            line = value.strip()
            # wikitext keeps its section headings; they are not prose.
            if not line or line.startswith("="):
                continue
            chunks.append(line)
            total += len(line)
            if total >= limit_chars:
                break
        return "\n".join(chunks)
    raise SystemExit("no corpus found: expected a cached wikitext-2 parquet")


def sentences(corpus: str) -> list[str]:
    out = []
    for para in corpus.split("\n"):
        for sent in re.split(r"(?<=[.!?])\s+", para):
            sent = sent.strip()
            if len(sent) > 40:
                out.append(sent)
    return out


def spans(sents: Sequence[str], n_letters: int, count: int,
          rng: random.Random) -> list[tuple[str, str]]:
    """(natural text, letters) pairs of at least `n_letters` letters.

    Cut at word boundaries, so the natural-spacing reading is real English
    rather than a text starting mid-word.

    An earlier version required the count to be exact. That silently selects
    for spans whose cumulative word lengths hit the target, which favours short
    words: mean word length was 4.82 in exact-length samples against 5.04 in
    unfiltered ones at the same target. Since the price is a difference between
    two per-letter figures over one letter sequence, the denominator only has
    to match within a span, not across them, so the requirement bought nothing
    and cost a biased sample. Each span now carries its own letter count.
    """
    out: list[tuple[str, str]] = []
    pool = list(sents)
    rng.shuffle(pool)
    for sent in pool:
        words = sent.split()
        start = 0
        while start < len(words):
            taken: list[str] = []
            letters = ""
            i = start
            while i < len(words) and len(letters) < n_letters:
                piece = normalize(words[i])
                if piece:
                    taken.append(words[i])
                    letters += piece
                i += 1
            if len(letters) >= n_letters and taken:
                out.append((" ".join(taken), letters))
                if len(out) >= count:
                    return out
            start = i if i > start else start + 1
    return out


# ---------------------------------------------------------------- experiment

class Row(NamedTuple):
    # NamedTuple rather than a dataclass: `@dataclass` inspects
    # sys.modules[cls.__module__], which is None when a file is loaded by
    # spec without being registered, and tests/test_docs.py loads every
    # experiment that way to check it still imports.
    model: str
    strategy: str
    n_letters: int
    n_spans: int
    n_scored: int
    coverage_forward: float
    coverage_reversed: float
    bits_natural: float
    bits_forward: float
    bits_reversed: float
    mirror_cost: float
    mirror_cost_sd: float
    mirror_cost_se: float
    thinning_per_letter: float


def measure(scorer: Scorer, sents: Sequence[str], vocab: frozenset[str],
            n_letters: int, count: int, strategy: str,
            rng: random.Random, show: int = 0,
            min_rev_coverage: float = 0.0) -> Row:
    sample = spans(sents, n_letters, count, rng)
    cov_f: list[float] = []
    cov_r: list[float] = []
    nat_bits: list[float] = []
    fwd_bits: list[float] = []
    rev_bits: list[float] = []
    deltas: list[float] = []
    shown = 0

    for natural, letters in sample:
        seg_f = segment(letters, vocab, strategy)
        seg_r = segment(letters[::-1], vocab, strategy)
        c_f = coverage(seg_f, vocab)
        c_r = coverage(seg_r, vocab)
        if c_r < min_rev_coverage:
            continue
        cov_f.append(c_f)
        cov_r.append(c_r)

        lp_nat = scorer.total_logprob(natural)
        lp_fwd = scorer.total_logprob(" ".join(seg_f))
        lp_rev = scorer.total_logprob(" ".join(seg_r))
        if lp_nat is None or lp_fwd is None or lp_rev is None:
            continue

        n = len(letters)
        b_nat = -lp_nat / NATS_PER_BIT / n
        b_fwd = -lp_fwd / NATS_PER_BIT / n
        b_rev = -lp_rev / NATS_PER_BIT / n
        nat_bits.append(b_nat)
        fwd_bits.append(b_fwd)
        rev_bits.append(b_rev)
        deltas.append(b_rev - b_fwd)

        if shown < show:
            print(f"    fwd: {' '.join(seg_f)}")
            print(f"    rev: {' '.join(seg_r)}")
            print(f"         {b_fwd:.2f} -> {b_rev:.2f} bits/letter")
            shown += 1

    if not deltas:
        raise SystemExit(f"no span of {n_letters} letters could be scored")

    delta = statistics.fmean(deltas)
    sd = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
    return Row(
        model=scorer.name,
        strategy=strategy,
        n_letters=n_letters,
        n_spans=len(sample),
        n_scored=len(deltas),
        coverage_forward=statistics.fmean(cov_f),
        coverage_reversed=statistics.fmean(cov_r),
        bits_natural=statistics.fmean(nat_bits),
        bits_forward=statistics.fmean(fwd_bits),
        bits_reversed=statistics.fmean(rev_bits),
        mirror_cost=delta,
        mirror_cost_sd=sd,
        mirror_cost_se=sd / math.sqrt(len(deltas)),
        thinning_per_letter=2.0 ** delta,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=["gpt2"])
    ap.add_argument("--spans", nargs="+", type=int,
                    default=[20, 30, 40, 60, 80, 120])
    ap.add_argument("--strategies", nargs="+",
                    default=["unigram", "fewest", "greedy"])
    ap.add_argument("--n", type=int, default=150, help="spans per cell")
    ap.add_argument("--vocab", default="data/lexicon.txt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show", type=int, default=0, help="print N examples")
    ap.add_argument("--min-reversed-coverage", type=float, default=0.0,
                    help="keep only spans whose REVERSED reading places at "
                         "least this fraction of its letters in real words. "
                         "The check that the price is not an artifact of "
                         "single-letter units: if it survives here, it is "
                         "about the words and not about the fallback.")
    ap.add_argument("--out", default="experiments/mirror_cost.json")
    args = ap.parse_args()

    vocab = load_vocab(args.vocab)
    corpus = load_corpus()
    sents = sentences(corpus)
    print(f"vocabulary {len(vocab)} words, corpus {len(sents)} sentences\n")

    rows: list[Row] = []
    for model_name in args.models:
        scorer = Scorer(model_name)
        for strategy in args.strategies:
            for n_letters in args.spans:
                rng = random.Random(args.seed)
                row = measure(scorer, sents, vocab, n_letters, args.n,
                              strategy, rng, show=args.show,
                              min_rev_coverage=args.min_reversed_coverage)
                rows.append(row)
                print(f"{row.model:22s} {row.strategy:8s} L={row.n_letters:4d} "
                      f"n={row.n_scored:4d} "
                      f"cov={row.coverage_forward:.2f}/{row.coverage_reversed:.2f} "
                      f"nat={row.bits_natural:5.2f} "
                      f"fwd={row.bits_forward:5.2f} "
                      f"rev={row.bits_reversed:5.2f} "
                      f"cost={row.mirror_cost:5.2f} "
                      f"+-{row.mirror_cost_se:.2f}")
        del scorer
        print()

    with open(args.out, "w") as fh:
        json.dump([r._asdict() for r in rows], fh, indent=2)
    print(f"wrote {args.out}")

    costs = [r.mirror_cost for r in rows]
    print(f"\nmirror cost across {len(rows)} cells: "
          f"{min(costs):.2f} to {max(costs):.2f} bits per free letter "
          f"(mean {statistics.fmean(costs):.2f})")


if __name__ == "__main__":
    main()

"""One generation of the phrase-inventory loop.

Run by the loop driver, once per configuration. It generates `--samples`
palindromes, measures them, and writes a JSON record including the sentences a
judge should be shown.

The coherence gain is recorded but NEVER optimised against: it is gameable 8x
by making the tail repeat the head (`coherence_gameability.py`), and this loop
tunes the inventory rather than the metric. What the loop actually steers by is
the judge's verdict and attested-bigram coverage, neither of which the search
can see.

The judge batch is BLINDED and salted with controls. Real English sentences and
word-salad sentences go in unlabelled alongside the palindrome's, so a judge
that passes everything or rejects everything is caught by its own answers
rather than trusted. `judge_key` records which is which for scoring afterwards.

    python experiments/phrase_loop.py --phrase-weight 0.0 --top-n 20000
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
from llm_palindrome.phrases import build_inventory, build_units
from llm_palindrome.scoring import CoherentScorer
from llm_palindrome.search import WordTries
from llm_palindrome.textify import (segment_at_units, segment_at_weak_joins,
                                    textify)
from llm_palindrome.validator import is_palindrome, normalize

_CACHE: dict = {}


def load_ngrams(orders: str, per_order: int) -> list[str]:
    """Clause-length units mined from running text.

    `data/ngrams_wikitext2.json` holds n-grams that occur at least twice in
    wikitext, so each one is a fragment somebody actually wrote rather than an
    inference from pair counts. Placing one whole puts that fragment in the
    output intact — which is the only mechanism here that can put a run of six
    grammatical words in a row into a palindrome.
    """
    if not orders:
        return []
    data = json.loads(Path("data/ngrams_wikitext2.json").read_text())
    out = []
    for order in orders.split(","):
        out.extend(data.get(order.strip(), [])[:per_order])
    return out


def resources(top_n: int, ngram_orders: str = "", per_order: int = 4000):
    """Vocabulary, bigrams and tries, cached across configs in one process."""
    if "vocab" not in _CACHE:
        _CACHE["vocab"] = build_vocab()
        _CACHE["bigrams"] = BigramModel.from_file("data/count_2w.txt",
                                                  vocab=set(_CACHE["vocab"]))
        _CACHE["pairs"] = None
    vocab, bg = _CACHE["vocab"], _CACHE["bigrams"]
    key = f"tries:{top_n}:{ngram_orders}:{per_order}"
    if key not in _CACHE:
        inv = []
        if top_n:
            inv += build_inventory("data/count_2w.txt", vocab=vocab, top_n=top_n)
        inv += load_ngrams(ngram_orders, per_order)
        tries = WordTries(build_units(vocab, inv) if inv else vocab)
        _CACHE[key] = (tries, inv)
    return vocab, bg, _CACHE[key]


def coverage(units: list[str], bg) -> float:
    """Share of adjacent word pairs that are attested — the service's number."""
    flat = [w for u in units for w in u.split()]
    if len(flat) < 2:
        return 0.0
    hits = sum(1 for a, b in zip(flat, flat[1:]) if bg._fwd.get(a, {}).get(b))
    return round(hits / (len(flat) - 1), 3)


def is_clean_sentence(text: str) -> bool:
    """Is this a well-formed sentence, fit to test a judge with?

    Generation 6's calibration failed on 3 of 5 real controls and the judge was
    right each time — the sampler was handing it truncated wikitext with stray
    brackets, unbalanced quotes and dangling numerals. A control that is not a
    sentence proves nothing about a judge that rejects it.
    """
    import re
    text = text.strip()
    if not (6 <= len(text.split()) <= 14):
        return False
    if not text[:1].isupper():
        return False
    if not text.endswith((".", "!", "?")):
        return False
    if re.search(r"[\[\]()<>\"\u2018\u2019\u201c\u201d]|\.\.\.|\u2026", text):
        return False
    if re.search(r"\d", text):
        return False
    if re.search(r"[.!?]", text[:-1]):     # a terminator anywhere but the end
        return False
    return True


def control_sentences(n: int, seed: int) -> tuple[list[str], list[str]]:
    """Real English sentences and word-salad ones, for blinding the judge."""
    from experiments.coherence_calibration import clean, load_paragraphs
    import re
    paras = [clean(p) for p in load_paragraphs(n * 12, 40, seed)]
    sents = [s.strip() for p in paras
             for s in re.split(r"(?<=[.!?]) +", p) if is_clean_sentence(s.strip())]
    rng = random.Random(seed)
    rng.shuffle(sents)
    real = sents[:n]
    salad = []
    for s in sents[n:n * 2]:
        ws = s.split()
        rng.shuffle(ws)
        salad.append(" ".join(ws))
    return real, salad


def _sentences(words, bg, args, units=None) -> list[str]:
    if args.segment == "units" and units is not None:
        segs = segment_at_units(units, min_unit_words=3)
        return textify(words, segments=segs).split(". ")
    if args.segment == "weak-joins":
        n = max(1, len(words) // max(1, args.words_per_sentence))
        segs = segment_at_weak_joins(words, bg, sentences=n)
        return textify(words, segments=segs).split(". ")
    return textify(words, args.words_per_sentence).split(". ")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phrase-weight", type=float, default=1.0)
    ap.add_argument("--top-n", type=int, default=20000,
                    help="inventory size; 0 disables phrases (the baseline)")
    ap.add_argument("--ngram-orders", default="",
                    help="comma-separated n-gram lengths to mine in, e.g. 4,5,6")
    ap.add_argument("--per-order", type=int, default=4000,
                    help="most frequent units to take from each order")
    ap.add_argument("--max-overhang", type=int, default=24,
                    help="letter debt cap")
    ap.add_argument("--min-letters", type=int, default=200)
    ap.add_argument("--long-bonus", type=float, default=0.0,
                    help="score per extra word in a unit")
    ap.add_argument("--freq-weight", type=float, default=0.25)
    ap.add_argument("--length-weight", type=float, default=0.12)
    ap.add_argument("--words-per-sentence", type=int, default=7)
    ap.add_argument("--segment", choices=["stride", "weak-joins", "units"], default="stride",
                    help="stride: cut every N words. weak-joins: cut where the "
                         "bigram model likes the join least.")
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--beam", type=int, default=60)
    ap.add_argument("--budget", type=float, default=8.0)
    ap.add_argument("--diversity", type=float, default=1.0)
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--judge-sentences", type=int, default=3,
                    help="sentences per sample offered to the judge")
    ap.add_argument("--no-coherence", action="store_true",
                    help="skip the GPT-2 diagnostic (faster iterations)")
    ap.add_argument("--label", default="gen")
    ap.add_argument("--out", default="runs/phrase_loop/gen.json")
    args = ap.parse_args()

    vocab, bg, (tries, inventory) = resources(args.top_n, args.ngram_orders,
                                              args.per_order)

    samples, t0 = [], time.time()
    for i in range(args.samples):
        seed = args.seed_base + i
        scorer = CoherentScorer(bg, freq_weight=args.freq_weight,
                                length_weight=args.length_weight,
                                phrase_weight=args.phrase_weight,
                                long_bonus=args.long_bonus)
        units = centerout_search(
            tries, scorer, min_letters=args.min_letters, beam_width=args.beam,
            seed=seed, max_steps=10**6, maximize="letters",
            candidate_limit=800, deadline=time.monotonic() + args.budget,
            diversity=args.diversity, max_overhang=args.max_overhang)
        if not units:
            continue
        text = " ".join(units)
        if not is_palindrome(text):
            continue
        words = [w for u in units for w in u.split()]
        samples.append({
            "seed": seed,
            "letters": len(normalize(text)),
            "units": len(units),
            "words": len(words),
            "phrases_used": sum(1 for u in units if " " in u),
            "longest_unit": max((len(u.split()) for u in units), default=0),
            "coverage": coverage(units, bg),
            "valid": True,
            "text": text,
            "sentences": _sentences(words, bg, args, units),
        })

    record = {
        "label": args.label,
        "config": {k: v for k, v in vars(args).items() if k not in ("out", "label")},
        "inventory_size": len(inventory),
        "closed": f"{len(samples)}/{args.samples}",
        "seconds": round(time.time() - t0, 1),
    }

    if samples:
        for field in ("letters", "coverage", "phrases_used", "words", "longest_unit"):
            record[field] = round(statistics.mean(s[field] for s in samples), 3)

    if samples and not args.no_coherence:
        cond = GPT2ConditionalScorer("gpt2", device="cpu")
        metric = CoherenceMetric(cond, controls=["unused"], skip_tokens=5)
        ctrl = SelfShuffledControls(n=6, seed=0)
        gains = []
        for s in samples:
            head, _ = split_at_word(s["text"])
            g = metric.score(s["text"], controls=ctrl(head)).gain
            if g is not None:
                s["coherence"] = round(g, 4)
                gains.append(g)
        if gains:
            record["coherence"] = round(statistics.mean(gains), 4)

    # The blinded judge batch: palindrome sentences mixed with real English and
    # word salad, shuffled together, keyed separately.
    rng = random.Random(args.seed_base)
    items = []
    for s in samples:
        for sent in s["sentences"][:args.judge_sentences]:
            if len(sent.split()) >= 4:
                items.append(("palindrome", s["seed"], sent.strip().rstrip(".") + "."))
    n_ctrl = max(2, len(items) // 4)
    real, salad = control_sentences(n_ctrl, args.seed_base)
    items += [("real", -1, s) for s in real] + [("salad", -1, s) for s in salad]
    rng.shuffle(items)

    record["judge_batch"] = [{"id": i, "text": t} for i, (_, _, t) in enumerate(items)]
    record["judge_key"] = [{"id": i, "kind": k, "seed": sd}
                           for i, (k, sd, _) in enumerate(items)]
    record["samples"] = samples

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(record, indent=2))

    print(f"{args.label}: closed={record['closed']} "
          f"letters={record.get('letters')} coverage={record.get('coverage')} "
          f"phrases={record.get('phrases_used')} "
          f"longest={record.get('longest_unit')} "
          f"coherence={record.get('coherence')} ({record['seconds']}s)")
    print(f"  inventory={record['inventory_size']} judge_items={len(items)}")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()

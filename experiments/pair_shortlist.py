"""Cut a hunt's output down to something a person can actually read.

`pair_hunt.py` returns thousands of letter-valid pairs ranked by GPT-2. That
ranking is a filter and not a verdict — the project's record on proxies is four
disagreements with blind judging and no agreements — so the last step before a
pair is allowed into `data/novel_pairs.json` is a human reading it.

The shortlist is ordered by a second, independent signal: how many of the joins
in each half are word pairs English has actually been seen to make. A half
whose every join is attested is not guaranteed to read, but a half with none
attested reliably does not, and the two signals disagree often enough that
ranking by both surfaces different material than ranking by either.

    python experiments/pair_shortlist.py --in runs/pair_hunt.json --top 200
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_palindrome.mining import attested_bigrams


def attested_fraction(words, attested) -> float:
    """Share of this half's joins that are attested. One word has no joins."""
    joins = list(zip(words, words[1:]))
    if not joins:
        return 0.0
    return sum((a, b) in attested for a, b in joins) / len(joins)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="src", default="runs/pair_hunt.json")
    ap.add_argument("--bigrams", default="data/count_2w.txt")
    ap.add_argument("--top", type=int, default=200)
    ap.add_argument("--min-attested", type=float, default=0.0,
                    help="floor on the WEAKER half's attested-join share")
    ap.add_argument("--model", default="",
                    help="rescore with a larger LM than the hunt used. "
                         "gpt2-large ranks these visibly better than gpt2 and "
                         "is far too slow to run inside the walk.")
    ap.add_argument("--rescore", type=int, default=4000,
                    help="how many of the hunt's own ranking to rescore")
    ap.add_argument("--sentence-like", action="store_true",
                    help="keep only pairs whose BOTH halves have a tag "
                         "reading with a subject and a verb (see syntax.py). "
                         "Narrows what a person reads; does not decide.")
    ap.add_argument("--order-gain", type=int, default=0, metavar="N",
                    help="rank by score minus the mean over N shuffles of the "
                         "same words, which is what the ORDER bought. 0 uses "
                         "the raw score, which rewards common vocabulary.")
    ap.add_argument("--per-family", type=int, default=1,
                    help="siblings of one mirror core to print; 0 for all")
    ap.add_argument("--readings", type=int, default=4,
                    help="alternative segmentations of each half to offer; "
                         "the letters are fixed, the spelling is not")
    ap.add_argument("--out", default="runs/pair_shortlist.json")
    args = ap.parse_args()

    rows = json.loads(Path(args.src).read_text())["results"]
    attested = attested_bigrams(args.bigrams)

    if args.model:
        from llm_palindrome.lm_scoring import GPT2Scorer
        rows = rows[:args.rescore]
        lm = GPT2Scorer(args.model)
        score_texts = lambda texts: lm.score_texts(texts, batch_size=32)

        if args.order_gain:
            # Score each half against its own shuffles, so the part of the
            # score its vocabulary was earning cancels. See wordorder.py.
            from llm_palindrome.wordorder import rank_halves
            halves = [row[side] for row in rows for side in ("left", "right")]
            gains = rank_halves(halves, score_texts, n=args.order_gain)
            for i, row in enumerate(rows):
                a, b = gains[2 * i], gains[2 * i + 1]
                row["score"] = round(min(a, b), 4)
                row["mean"] = round((a + b) / 2, 4)
        else:
            texts = [" ".join(row[side]).capitalize() + "."
                     for row in rows for side in ("left", "right")]
            scores = score_texts(texts)
            for i, row in enumerate(rows):
                a, b = scores[2 * i], scores[2 * i + 1]
                row["score"] = round(min(a, b), 4)
                row["mean"] = round((a + b) / 2, 4)
        rows.sort(key=lambda r: -r["score"])

    for row in rows:
        left = attested_fraction(row["left"], attested)
        right = attested_fraction(row["right"], attested)
        row["attested"] = round(min(left, right), 3)
        row["attestedMean"] = round((left + right) / 2, 3)

    if args.sentence_like:
        # Both halves have to be readable as sentences with a subject and a
        # verb. On 179 walked pairs this keeps 3, which is the point: reading
        # 179 to find nothing is how the last four shortlists went.
        from llm_palindrome.syntax import brown_tables, sentence_like
        table, shapes, _ = brown_tables()
        for row in rows:
            row["sentenceLike"] = bool(
                sentence_like(row["left"], table, shapes)
                and sentence_like(row["right"], table, shapes))
        rows = [r for r in rows if r["sentenceLike"]]

    kept = [r for r in rows if r["attested"] >= args.min_attested]
    kept.sort(key=lambda r: (-r["attested"], -r.get("score", 0.0)))

    if args.per_family:
        # A shortlist is read top to bottom, and siblings sort together: eight
        # variants of one mirror core fill the first screen and none of the
        # rest is ever seen. Capping here rather than in the walk keeps the
        # walk's output complete.
        from collections import Counter

        from llm_palindrome.pairs import junction
        seen: Counter = Counter()
        spread = []
        for row in kept:
            fam = junction(row["left"], row["right"])
            if seen[fam] >= args.per_family:
                continue
            seen[fam] += 1
            spread.append(row)
        kept = spread

    kept = kept[:args.top]

    if args.readings > 1:
        # The walk returns ONE segmentation of each half, and the letters admit
        # others. Which one is offered decides whether a pair looks like a find
        # or like junk — "no sawn am a" and "no saw nam a" are the same letters
        # — so the shortlist carries the alternatives rather than the walk's
        # first guess.
        from llm_palindrome.generate import build_vocab
        from llm_palindrome.lexicon import is_real_word, load_lexicon
        from llm_palindrome.respace import respace_k
        from llm_palindrome.spelling import CONTRACTIONS

        lexicon = load_lexicon("data/lexicon.txt")
        vocab = ({w for w in build_vocab(60000) if is_real_word(w, lexicon)}
                 | set(CONTRACTIONS))
        for row in kept:
            for side in ("left", "right"):
                letters = "".join(row[side])
                other = [r for r in respace_k(letters, vocab, k=args.readings)
                         if r != row[side]]
                if other:
                    row[f"{side}Readings"] = [" ".join(r) for r in other]

    Path(args.out).write_text(json.dumps(kept, indent=1))
    print(f"{len(rows)} pairs -> {len(kept)} shortlisted -> {args.out}\n")
    for row in kept:
        print(f"  {row['attested']:.2f} {row.get('score', 0):7.3f}  "
              f"{' '.join(row['left'])} || {' '.join(row['right'])}")
        for side in ("left", "right"):
            for alt in row.get(f"{side}Readings", [])[:2]:
                print(f"        {side[0]}: {alt}")


if __name__ == "__main__":
    main()

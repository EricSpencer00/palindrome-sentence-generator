"""Build the unit inventory v2 searches over.

v2's trie holds whole sentences alongside single words, and this is what
produces them. The artifact is a corpus derivative, so it is staged rather than
committed — same as `data/count_2w.txt` — and this script is how a machine that
needs it gets one.

Two kinds of unit come out, and the difference matters:

  n-grams    spans from the middle of sentences, kept when they RECUR. Repetition
             is the only evidence that the words belong together.
  sentences  whole sentences, start to end, kept on a single occurrence. A
             sentence needs no corroboration to be a sentence.

Only the second produces text a judge accepts. Isolating an n-gram as a
sentence yields "Was unable to make." — grammatical, and trailing off.

    python training/build_inventory.py --out data/ngrams_wikitext2.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_palindrome.generate import build_vocab
from llm_palindrome.phrases import mine_ngrams, mine_sentences


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="wikitext")
    ap.add_argument("--config", default="wikitext-2-raw-v1")
    ap.add_argument("--split", default="train")
    ap.add_argument("--ngram-orders", default="3,4,5,6")
    ap.add_argument("--min-count", type=int, default=2,
                    help="occurrences before an n-gram is trusted as a unit")
    ap.add_argument("--sentence-caps", default="6,8,10",
                    help="word-count caps to bucket whole sentences by")
    ap.add_argument("--vocab", type=int, default=30000)
    ap.add_argument("--out", default="data/ngrams_wikitext2.json")
    args = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset(args.dataset, args.config, split=args.split)
    lines = [t for t in ds["text"] if t.strip() and not t.strip().startswith("=")]
    print(f"{len(lines)} lines, {sum(len(l.split()) for l in lines)} words")

    # The same filter the search uses. The frequency list carries slurs, and a
    # corpus-derived inventory is a second door into the same vocabulary.
    vocab = set(build_vocab(args.vocab))

    out: dict[str, list[str]] = {}
    for order in args.ngram_orders.split(","):
        n = int(order)
        out[str(n)] = mine_ngrams(lines, n=n, min_count=args.min_count, vocab=vocab)
        print(f"  {n}-grams (>={args.min_count}x): {len(out[str(n)])}")

    for cap in args.sentence_caps.split(","):
        c = int(cap)
        out[f"sent{c}"] = mine_sentences(lines, min_words=3, max_words=c, vocab=vocab)
        print(f"  sentences <={c} words: {len(out[f'sent{c}'])}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

"""Would the catalogue survive our own filters? Mostly not.

A filter is worth what it does to the material you are trying to find, and the
material is known: 29 catalogued mirror-pairs whose two halves both read. They
are the existence proof for the whole construction, so any test that rejects
them is rejecting the target.

Three tests, run against those 29:

  count_2w.txt      the attested-join constraint the walk was given
  Brown bigrams     a million words of edited prose
  GPT-2 top-400     a model's followers for each word, rank-thresholded
                    (needs data/joins_gpt2.json; the row is skipped without it)

The numbers are the point, and they agree with each other:

    count_2w.txt     joins 0.41    halves fully covered 0.27
    brown bigrams    joins 0.39    halves fully covered 0.29
    gpt-2 top 400    joins 0.38    halves fully covered 0.27

A PAIR needs both halves covered, so a hard constraint keeps something like 7%
of the target. "Evil rats on || no star live" has not one join in any of the
three. Neither does "faced no devil". These are not strange sentences; they are
ordinary English that a frequency list does not happen to record, because a
palindrome's English is odd English and frequency lists are made of the usual.

So attestation is a FREQUENCY test wearing a grammaticality test's clothes, and
a walk constrained by it steers away from the region the catalogue lives in.
The rank-thresholded model does no better, for a reason
that is not fixable by raising the threshold: after a one-word context the
distribution is broad, so "no star" is a fine continuation that sits below
thousands of other fine continuations.

What this rules out is a cheap LOCAL test for readability. What reads is a
property of the whole half, and the filters that work on whole halves — a
language model's score, a person — cannot be run inside the walk.

    python experiments/join_attestation.py [--joins data/joins_gpt2.json]
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def coverage(halves, allowed) -> tuple[float, float]:
    """(mean share of joins allowed, share of halves entirely allowed)."""
    shares = []
    for half in halves:
        joins = list(zip(half, half[1:]))
        if joins:
            shares.append(sum(allowed(a, b) for a, b in joins) / len(joins))
    if not shares:
        return 0.0, 0.0
    return statistics.mean(shares), sum(s == 1.0 for s in shares) / len(shares)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--units", default="data/mirror_units.json")
    ap.add_argument("--bigrams", default="data/count_2w.txt")
    ap.add_argument("--joins", default="data/joins_gpt2.json",
                    help="model join table; skipped when absent")
    args = ap.parse_args()

    from llm_palindrome.mining import attested_bigrams

    units = json.loads(Path(args.units).read_text())
    halves = [u[side] for u in units for side in ("left", "right")]
    print(f"{len(units)} catalogued pairs, {len(halves)} halves\n")

    web = attested_bigrams(args.bigrams)
    mean, whole = coverage(halves, lambda a, b: (a, b) in web)
    print(f"  count_2w.txt    joins {mean:.2f}   halves fully covered {whole:.2f}")

    try:
        from nltk.corpus import brown
        words = [w.lower() for w in brown.words() if w.isalpha()]
        pairs = set(zip(words, words[1:]))
        mean, whole = coverage(halves, lambda a, b: (a, b) in pairs)
        print(f"  brown bigrams   joins {mean:.2f}   halves fully covered {whole:.2f}")
    except LookupError:
        print("  brown bigrams   (corpus not installed)")

    table_path = Path(args.joins)
    if table_path.exists():
        table = {k: set(v) for k, v in json.loads(table_path.read_text()).items()}
        mean, whole = coverage(halves, lambda a, b: b in table.get(a, ()))
        print(f"  gpt-2 followers joins {mean:.2f}   halves fully covered {whole:.2f}")
    else:
        print(f"  gpt-2 followers (no {table_path}; "
              f"build with training/build_join_table.py)")

    print("\nEvery number above is the share of the TARGET a filter would keep.")


if __name__ == "__main__":
    main()

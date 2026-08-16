"""Author one half and let the letters decide the other. It does not work.

The tempting division of labour: a model writes the left half, which is
therefore grammatical English, and segmentation recovers the right half from
its reversed letters. It puts the hard part — writing — where the writing is
good, and leaves the arithmetic to the machine.

Measured over 148 authored sentences in `data/authored_sentences.txt`, three to
six words each, ordinary vocabulary:

    148 authored          12 have any spellable mirror (8.1%)
     12 spellable          0 whose mirror reads

and the zero is not close. What comes back is "no to lad ah i", "eta gat a saw
i", "moo rani saw i" — the reversed letters of an English sentence are not
English, and choosing which English sentence to write barely moves that.

The reason is visible in the failures. A left half beginning "i" or "a" forces
the right half to END on "i" or "a", and a left half beginning "the" forces it
to end on "eht", which is not a word at all. The opening word of one half is
the closing word of the other, and authoring picks it for the wrong end.

So both halves have to be chosen together, which is what `llm_palindrome/pairs.py`
does and why the walk is the material engine rather than a fallback.

    python experiments/authored_mirrors.py
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="src", default="data/authored_sentences.txt")
    ap.add_argument("--readings", type=int, default=6)
    args = ap.parse_args()

    from llm_palindrome.generate import build_vocab
    from llm_palindrome.lexicon import is_real_word, load_lexicon
    from llm_palindrome.respace import respace_k
    from llm_palindrome.spelling import CONTRACTIONS

    lexicon = load_lexicon("data/lexicon.txt")
    vocab = ({w for w in build_vocab(60000) if is_real_word(w, lexicon)}
             | set(CONTRACTIONS) | {"a", "i"})
    vocab = {w for w in vocab if len(w) > 1 or w in ("a", "i")}

    lines = [l.strip().lower() for l in Path(args.src).read_text().splitlines()]
    lines = [l for l in lines if l]

    spellable = 0
    for phrase in lines:
        letters = "".join(c for c in phrase if c.isalpha())
        readings = respace_k(letters[::-1], vocab, k=args.readings)
        if not readings:
            continue
        spellable += 1
        print(f"{phrase}")
        for reading in readings:
            print(f"       {' '.join(reading)}")

    print(f"\n{len(lines)} authored, {spellable} with a spellable mirror "
          f"({spellable / max(1, len(lines)):.1%})")
    print("Read them: the count of mirrors that READ is the number that matters.")


if __name__ == "__main__":
    main()

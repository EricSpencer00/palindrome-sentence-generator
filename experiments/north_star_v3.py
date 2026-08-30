"""The six mechanical north-star criteria, applied to v3 and to v2 as control.

`tests/test_north_star.py` runs these checks against the v2 paragraph endpoint,
which is what they were written for. v3 now carries the name the north star
reserved, so the same checks have to be run against it and the answer published
whichever way it comes out.

Criteria 6, 7 and 8 — grammatical, has a subject, reads as coherent prose —
need blind judging against salad and real-prose controls. They are not
computed here and nothing this script prints is evidence about them.

    python experiments/north_star_v3.py [--seeds 24]

Writes nothing; the table it prints is the table in RESULTS-north-star-v3.md.
"""
from __future__ import annotations

import argparse
import collections
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CRITERIA = (1, 2, 3, 4, 5, 9)
LENGTHS = (400, 1200, 4000, 14500)


def sentences_of(text: str) -> list[str]:
    return [s.strip() for s in text.split(".") if s.strip()]


def letters_of(text: str) -> str:
    """Not `validator.normalize`. Criterion 2 is checked without our own code,
    because our own code is the thing under test."""
    return "".join(c.lower() for c in text if c.isalpha())


def check(text: str) -> dict[int, bool]:
    from llm_palindrome.paragraphs import is_novel_palindrome
    from llm_palindrome.validator import is_palindrome

    sents = sentences_of(text)
    said = [s.lower() for s in sents]
    half = len(said) // 2
    lets = letters_of(text)
    return {
        1: len(re.findall(r"[A-Za-z]+", text)) >= 100,
        2: lets == lets[::-1],
        3: len([s for s in sents if is_palindrome(s)]) <= 1,
        4: len(set(said)) == len(said),
        5: not (set(said[:half]) & set(said[half + 1:])),
        9: all(is_novel_palindrome(s) for s in sents) and is_novel_palindrome(text),
    }


def tally(rows: list[dict[int, bool]]) -> str:
    n = len(rows)
    return "  ".join(f"{sum(r[k] for r in rows):>2}/{n}" for k in CRITERIA)


def repeated_sentences(text: str) -> list[tuple[str, int]]:
    said = [s.lower() for s in sentences_of(text)]
    return sorted(((s, n) for s, n in collections.Counter(said).items() if n > 1),
                  key=lambda t: -t[1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=24)
    args = ap.parse_args()

    import server.v3 as v3

    head = "  ".join(f"  c{k}" for k in CRITERIA)
    print(f"v3 /composition, novel=true, {args.seeds} seeds per length")
    print(f"  {'letters':>8}  {head}")
    for letters in LENGTHS:
        rows = [check(v3.composition(seed=s, letters=letters, chops=None,
                                     longest_first=False, centre="", novel=True)["text"])
                for s in range(args.seeds)]
        print(f"  {letters:>8}  {tally(rows)}")

    # Why criterion 4 fails, stated as the thing it actually is: the chunk
    # guarantee holds and the sentence cut collides.
    print("\nCriterion 4, in detail — the endpoint's own repeat count is the "
          "CHUNK count, and it is zero throughout")
    for letters in LENGTHS[:3]:
        comp = v3.composition(seed=0, letters=letters, chops=None,
                              longest_first=False, centre="", novel=True)
        dup = repeated_sentences(comp["text"])
        print(f"  letters={letters:>6}  chunks_repeated={comp['repeats']}  "
              f"sentence_types_repeated={len(dup)}  "
              f"worst={dup[0] if dup else '-'}")

    print(f"\nCONTROL — v2 /paragraph, which is what tests/test_north_star.py "
          f"measures, {args.seeds} draws")
    from server.v2 import letter_paragraph
    rows = [check(letter_paragraph(sentences=9)["text"]) for _ in range(args.seeds)]
    print(f"  {'':>8}  {tally(rows)}")


if __name__ == "__main__":
    main()

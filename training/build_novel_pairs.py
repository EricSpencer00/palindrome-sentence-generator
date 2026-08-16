"""Turn chosen mirror-pairs into the bank the endpoint serves.

    PYTHONPATH=. python3 training/build_novel_pairs.py runs/pair_chosen.json

The input is a list of pairs a person picked out of `pair_shortlist.py` output.
The picking is the part no script does: criteria 6-8 of docs/NORTH-STAR.md are
grammaticality, having a subject, and reading as prose, and four automated
proxies in this project have disagreed with blind judging on exactly those.

What this does is everything that CAN be checked, so a slip in the picking is
caught before it ships:

  mirror     the two halves spell each other backwards, letter for letter
  novel      absent from the catalogued palindrome record
  safe       no word the public vocabulary filter would have withheld
  distinct   no word used on both sides, no pair repeated
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

OUT = Path("data/novel_pairs.json")


def problems(left: list[str], right: list[str], seen: set,
             safe: set, known_ok) -> list[str]:
    """Everything wrong with one pair, so all of it is reported at once."""
    from llm_palindrome.validator import is_palindrome

    out = []
    if "".join(right) != "".join(left)[::-1]:
        out.append("halves do not mirror")
    # Criterion 3, enforced at the unit. A self-palindromic half spells itself
    # backwards, so its partner is the same letters and the mirror does no work
    # for that sentence — the refrain, one sentence at a time.
    for half in (left, right):
        if is_palindrome(" ".join(half)):
            out.append(f"half is its own palindrome: {' '.join(half)}")
    if not known_ok(" ".join(left + right)):
        out.append("catalogued")
    for word in left + right:
        if word not in safe:
            out.append(f"withheld word: {word}")
    if set(left) & set(right):
        out.append(f"word on both sides: {sorted(set(left) & set(right))}")
    key = (" ".join(left), " ".join(right))
    if key in seen:
        out.append("duplicate")
    seen.add(key)
    return out


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2

    from llm_palindrome.generate import build_vocab
    from llm_palindrome.paragraphs import is_novel_palindrome
    from llm_palindrome.spelling import CONTRACTIONS

    rows = json.loads(Path(argv[1]).read_text())
    # The endpoint's own vocabulary, so a pair cannot ship a word the public
    # generator would have refused to place.
    safe = set(build_vocab(60000)) | set(CONTRACTIONS)

    seen: set = set()
    kept, rejected = [], []
    for row in rows:
        left, right = list(row["left"]), list(row["right"])
        bad = problems(left, right, seen, safe, is_novel_palindrome)
        if bad:
            rejected.append((" ".join(left), " ".join(right), bad))
            continue
        kept.append({"left": left, "right": right,
                     "source": row.get("source", argv[1])})

    for l, r, bad in rejected:
        print(f"  rejected: {l} || {r}  ({'; '.join(bad)})")
    OUT.write_text(json.dumps(kept, indent=1) + "\n")
    words = sum(len(p["left"]) + len(p["right"]) for p in kept)
    print(f"{len(kept)} pairs kept, {len(rejected)} rejected -> {OUT}")
    print(f"they assemble into a {words}-word paragraph")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

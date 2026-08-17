"""Bank a few palindromes the search already closed, for when it does not.

    PYTHONPATH=. python3 training/build_fallbacks.py --count 12

A beam search on a deadline does not close every time. Answering "no palindrome
closed; try again" is honest and it is also a blank page, and the visitor did
not ask to be told about our beam — so the endpoint keeps a small bank and
serves one at random when a search comes back empty, labelled `fallback: true`
so nobody mistakes it for something written for them.

Every entry here is this project's own output: the same search, the same
vocabulary, run ahead of time instead of while someone waits. Nothing is
catalogued and nothing is authored, which is the only reason a canned answer is
allowed to appear on this endpoint at all.

They are stored as word lists rather than as finished payloads. The response
shape belongs to `_shape`, and a bank of frozen payloads would drift from it
the first time that function changed.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

OUT = Path("data/fallback_texts.json")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--count", type=int, default=12)
    ap.add_argument("--budget", type=float, default=12.0)
    ap.add_argument("--attempts", type=int, default=60,
                    help="searches to run; the ones that close are kept")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    os.environ["PALINDROME_NO_WARM"] = "1"
    import server.v2 as v2
    from llm_palindrome.validator import is_palindrome, normalize

    v2._warm()
    if v2._tries is None:
        raise SystemExit(f"inventory did not load: {v2._load_error}")

    kept: list[dict] = []
    seen: set[str] = set()
    for nonce in range(args.attempts):
        if len(kept) >= args.count:
            break
        found = v2._search("", args.budget, None, nonce=nonce)
        if found is None:
            continue
        words = list(found["left"]) + list(found["right"])
        text = " ".join(words)
        # The bank is served without being searched again, so it is checked
        # here rather than trusted: a broken entry would be a broken palindrome
        # shipped to every visitor whose search came back empty.
        if not is_palindrome(text) or normalize(text) in seen:
            continue
        seen.add(normalize(text))
        kept.append({"words": words, "letters": len(normalize(text))})
        print(f"  {len(kept):2}/{args.count}  {kept[-1]['letters']:5} letters  "
              f"{text[:60]}...")

    if not kept:
        raise SystemExit("nothing closed; the search is broken, not unlucky")

    Path(args.out).write_text(json.dumps(kept, indent=1) + "\n")
    print(f"{len(kept)} palindromes -> {args.out}")


if __name__ == "__main__":
    main()

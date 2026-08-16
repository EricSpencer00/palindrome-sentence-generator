"""Ship a headword list so mining does not depend on a system dictionary.

    PYTHONPATH=. python3 training/build_lexicon.py

Intersects a system dictionary (macOS/BSD ship web2 at /usr/share/dict/words)
with the frequency list, so the result is both a real word and one anybody
uses — web2 alone carries 234k entries, most of them archaic. The output is
committed; only this script needs the system file.
"""
from __future__ import annotations

import sys
from pathlib import Path

DICTIONARIES = ("/usr/share/dict/words", "/usr/share/dict/web2",
                "/usr/dict/words")
OUT = Path("data/lexicon.txt")


def system_dictionary() -> set[str]:
    for path in DICTIONARIES:
        if Path(path).exists():
            with open(path, encoding="utf-8", errors="ignore") as handle:
                return {line.strip().lower() for line in handle
                        if line.strip().isalpha()}
    raise SystemExit(
        "no system dictionary found; tried " + ", ".join(DICTIONARIES))


def main() -> int:
    from wordfreq import top_n_list

    headwords = system_dictionary()
    # 200k rather than the generator's 30k: the right half of a mined pair may
    # legitimately use a word the generator would never pick as a left half.
    frequent = [w for w in top_n_list("en", 200000)
                if w.isalpha() and w.isascii()]
    keep = sorted(w for w in frequent if w in headwords)

    OUT.write_text("\n".join(keep) + "\n")
    print(f"system dictionary {len(headwords)}, frequent {len(frequent)}"
          f" -> {len(keep)} shipped to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

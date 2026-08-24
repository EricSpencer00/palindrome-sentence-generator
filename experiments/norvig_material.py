"""Norvig's palindrome against this one, on the axis the two methods differ on.

Not length. norvig.com/palindrome.html is 90,439 letters and nothing here is
within an order of magnitude of it, because the two programs answer different
questions: his chooses the branch that lets the search keep going, this one
chooses the branch that reads. Comparing the lengths would be comparing the
questions.

What is comparable is the MATERIAL. Both programs spend a dictionary, and the
difference between the dictionaries is most of the difference between the
outputs, so: what fraction of the words on the page are words a reader accepts?
Measured with this project's own filters — `lexicon.is_real_word` against the
52,927-headword lexicon, and `shortwords.is_real_short` — applied to his text
and to ours alike.

    python experiments/norvig_material.py

Fetches two files from norvig.com on first run and caches them under
`runs/norvig/`. Prints the table in `RESULTS-norvig-material.md`.
"""
from __future__ import annotations

import html
import random
import re
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CACHE = Path("runs/norvig")
SOURCES = {
    # His version 3, the 21,012-word palindrome, and the word list behind it.
    "pal21txt.html": "https://www.norvig.com/pal21txt.html",
    "npdict.txt": "https://www.norvig.com/npdict.txt",
}
LENGTHS = (400, 1200, 4000)


def fetch(name: str) -> str:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / name
    if not path.exists():
        with urllib.request.urlopen(SOURCES[name], timeout=60) as r:
            path.write_bytes(r.read())
    return path.read_text(encoding="utf-8", errors="replace")


def tokens(text: str) -> list[str]:
    return [w.lower() for w in re.findall(r"[A-Za-z]+", text)]


def strip_tags(page: str) -> str:
    body = page.split("</h1>", 1)[1] if "</h1>" in page else page
    return html.unescape(re.sub(r"<[^>]+>", " ", body))


def profile(name: str, words: list[str], lex) -> None:
    from llm_palindrome.lexicon import is_real_word
    from llm_palindrome.shortwords import is_real_short

    words = [w for w in words if w]
    n = len(words)
    short = sum(1 for w in words if len(w) <= 2)
    real = sum(1 for w in words if is_real_word(w, lex))
    both = sum(1 for w in words if is_real_word(w, lex) and is_real_short(w))
    mean = sum(len(w) for w in words) / max(1, n)
    print(f"{name:44} {n:>7}  {short / n:6.1%}  {real / n:6.1%}  "
          f"{both / n:6.1%}  {mean:5.2f}")


def main() -> None:
    from llm_palindrome.lexicon import load_lexicon

    lex = load_lexicon("data/lexicon.txt")
    print(f"{'text':44} {'words':>7}  {'1-2ch':>6}  {'lexic':>6}  "
          f"{'both':>6}  {'mean':>5}")

    profile("Norvig v3, the 21,012-word palindrome",
            tokens(strip_tags(fetch("pal21txt.html"))), lex)
    profile("Norvig npdict.txt, as tokens",
            [t for line in fetch("npdict.txt").splitlines() for t in tokens(line)],
            lex)

    # The floor. A frequency-weighted draw rather than the raw vocabulary,
    # because what a reader meets is the weighted distribution — the raw list
    # is mostly long rare words and would put the bar in the wrong place.
    from wordfreq import top_n_list, word_frequency
    vocab = top_n_list("en", 30000)
    rng = random.Random(0)
    profile("English, frequency-weighted sample",
            rng.choices(vocab, weights=[word_frequency(w, "en") for w in vocab],
                        k=20000), lex)

    import server.v3 as v3
    v3.ensure_loaded()
    for target in LENGTHS:
        comp = v3.composition(seed=1, letters=target, chops=None,
                              longest_first=False, centre=None, novel=True)
        profile(f"here, v3 composition, {target} letters",
                tokens(comp["plain"]), lex)

    print("\nv1 is not run here: it needs the model and a live search. The row "
          "in RESULTS-norvig-material.md is one deployed run, 1,010 letters, "
          "taken from /api/generate.")


if __name__ == "__main__":
    main()

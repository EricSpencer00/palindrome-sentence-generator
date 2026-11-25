"""Build data/v5_examples.json: the paragraphs the page renders for readers
and crawlers. One process per length; best of `--seeds` by search quality.

    python -m server.v5_examples --beam 800 --seeds 3
"""
from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

JOBS = [(250, "sea,ship,storm"), (300, "night,moon,stars"), (400, "garden,rose,flower"),
        (500, "river,stone,bridge"), (600, "love,heart,tears"), (700, "war,blood,king"),
        (800, "winter,snow,cold"), (900, "god,heaven,soul"), (1000, "city,street,night"),
        (1200, "love,heart,tears")]
OUT = Path(__file__).resolve().parent.parent / "data" / "v5_examples.json"


def one(job):
    letters, theme, beam, seeds = job
    from llm_palindrome import paragraph as P
    from llm_palindrome.prose_lm import ProseLM, theme_bonus, theme_vectors
    from llm_palindrome.spectrum import Spectrum, Tables
    from llm_palindrome.validator import is_palindrome
    lm, tb = ProseLM.load(), Tables.load()
    lex, sp = P.Lexicon(lm, tb), Spectrum(tb)
    words = theme.split(",")
    best = None
    for seed in range(seeds):
        closed = P.grow(lm, lex, letters, beam=beam, spectrum=sp, sent_w=2.0, content_bonus=1.6,
                        theme=theme_bonus(theme_vectors(), words), theme_w=15.0, seed=seed, start_noise=4.0)
        if closed and (best is None or (closed[0].score + closed[0].adj) / len(closed[0].seq)
                       > (best.score + best.adj) / len(best.seq)):
            best = closed[0]
    text = P.render(best.seq)
    assert is_palindrome(text)
    return {"theme": words, "letters": len(re.sub("[^a-z]", "", text.lower())),
            "words": len(text.split()), "text": text}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--beam", type=int, default=800)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=5)
    a = ap.parse_args()
    # Build the shared caches once before forking workers.
    from llm_palindrome.prose_lm import ProseLM, theme_vectors
    from llm_palindrome.spectrum import Tables
    ProseLM.load(), Tables.load(), theme_vectors()
    with ProcessPoolExecutor(a.workers) as pool:
        rows = list(pool.map(one, [(n, t, a.beam, a.seeds) for n, t in JOBS]))
    OUT.write_text(json.dumps(rows, indent=1))
    for r in rows:
        print(f"\n[{', '.join(r['theme'])} | {r['letters']} letters, {r['words']} words]\n{r['text']}")


if __name__ == "__main__":
    main()

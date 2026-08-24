"""Grow a palindrome from the outside in, instead of searching for a longer one.

The idea is not mine. Shown `non academia aimed a canon` (22 letters), the
account holder wrote

    we racecar non academia anna aimed a canon racecar ew        (44 letters)
    we racecared non academia anna aimed a canon de racecar ew   (48 letters)

by hand, and both verify. That is twice the length of anything the search
produced today, obtained in one step, and it works because every operation used
preserves the mirror by construction rather than by search.

The operations
--------------
A palindrome's letters split at the midpoint into `L` and `reverse(L)`. Any of
these keeps that property:

    WRAP-PAIR   put word `w` at the front and `reverse(w)` at the back, where
                both are words.  we ... ew
    WRAP-SELF   put the same self-palindromic word at both ends.
                racecar ... racecar
    CENTRE      insert a self-palindromic word at the exact midpoint.
                ... anna ...
    GLUE        the account holder's second example: attach letters to the
                inside of a wrap so the join reads, and pay the debt with the
                mirrored letters on the other side.  racecar'ed ... de racecar

WRAP and CENTRE cost nothing: the debt each creates is discharged by its own
mirror in the same move. That is exactly why they succeed where the chunking
test failed. A chunk placed alone advances the mirror by 1.09 letters whatever
its size, because it leaves its own length behind as debt. A chunk placed
*together with its mirror* leaves no debt at all, and advances by its full
length on both sides.

So the measured conservation law is not a wall. It is a statement about placing
units one at a time, and these operations place them two at a time.

What is measured here
---------------------
Starting from every verified palindrome this project has — the catalogue and
today's Polaris output — apply the operations greedily and record how far each
seed goes, whether the result still passes the syntactic test it started with,
and what it reads like. Length is easy; the question is whether anything
survives the growth.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_palindrome.generate import build_vocab
from llm_palindrome.present import present
from llm_palindrome.syntax import brown_tables, sentence_like
from llm_palindrome.validator import is_palindrome, normalize


def material(vocab):
    """Self-palindromic words, and (w, reverse(w)) pairs that are both words."""
    s = set(vocab)
    selfpal = [w for w in vocab if len(w) >= 3 and w == w[::-1]]
    pairs = [(w, w[::-1]) for w in vocab
             if 2 <= len(w) and w[::-1] in s and w != w[::-1]]
    return selfpal, pairs


def glue_pairs(path: str = "data/mirror_pairs.json"):
    """Multi-word wraps: the `racecared` / `de racecar` shape, generalised.

    A single-word wrap needs `reverse(w)` to be one word too, which is a
    severe restriction — 598 pairs in a 30,000-word vocabulary. Dropping the
    requirement that the two sides have the same number of words opens it up:
    `an item` mirrors `met in a`, `warning is` mirrors `sign in raw`. The
    letters mirror; the word boundaries need not.

    That is what the account holder's second example does. `racecared` is one
    token and `de racecar` is two, and they mirror because the mirror has never
    cared where the spaces are.

    Attested pairs first: both halves being phrases English is recorded as
    making is the difference between a wrap that reads and a wrap that merely
    fits.
    """
    rows = json.load(open(path))
    out = []
    for r in rows:
        left, right = list(r["left"]), list(r["right"])
        if normalize(" ".join(left)) != normalize(" ".join(right))[::-1]:
            continue
        out.append((left, right, bool(r.get("attested"))))
    out.sort(key=lambda t: (not t[2], len(t[0]) + len(t[1])))
    return out


def wrap_glue(words, left, right):
    return list(left) + list(words) + list(right)


def wrap_pair(words, a, b):
    return [a] + list(words) + [b]


def wrap_self(words, w):
    return [w] + list(words) + [w]


def centre(words, w):
    """Insert `w` at the word boundary nearest the letter midpoint.

    Only legal when that boundary is the exact midpoint; otherwise the two
    halves are not each other's mirror across the insertion point and the
    result is not a palindrome. The caller checks with `is_palindrome`
    regardless, which is the only arbiter this project trusts.
    """
    letters = [len(x) for x in words]
    half = sum(letters) / 2
    run = 0
    for i, n in enumerate(letters):
        if run == half:
            return list(words[:i]) + [w] + list(words[i:])
        run += n
    return None


def edge_score(words, table, shapes) -> float:
    """Does the text still open and close like English after a wrap?

    Length is trivial to gain here and worth nothing on its own: a greedy
    first-fit rule stacked `and`/`dna` six deep and reported 87 letters. So a
    wrap has to justify itself at the two places it actually changed, which are
    the first few words and the last few.
    """
    from llm_palindrome.syntax import shaped
    score = 0.0
    for edge in (words[:4], words[-4:]):
        if len(edge) >= 3 and shaped(edge, table, shapes):
            score += 1.0
    return score


def grow(words, selfpal, pairs, table, shapes, max_ops=6, glue=()):
    """Apply operations while the mirror holds AND the edges still parse.

    Each wrap must not repeat a word already used as glue, and must not lower
    the edge score. Ties go to the shorter addition, because a wrap that buys
    two letters and keeps the reading is worth more than one that buys eight
    and does not.
    """
    cur = list(words)
    trail: list[str] = []
    used: set[str] = set()

    for w in selfpal:
        cand = centre(cur, w)
        if cand and is_palindrome(" ".join(cand)):
            cur = cand
            trail.append(f"centre:{w}")
            used.add(w)
            break

    base = edge_score(cur, table, shapes)
    for _ in range(max_ops):
        best = None
        for a, b in pairs:
            if a in used or b in used:
                continue
            cand = wrap_pair(cur, a, b)
            if not is_palindrome(" ".join(cand)):
                continue
            sc = edge_score(cand, table, shapes)
            if sc >= base:
                key = (-sc, 0, len(a))
                if best is None or key < best[0]:
                    best = (key, cand, f"pair:{a}/{b}", {a, b}, sc)
        for w in selfpal:
            if w in used:
                continue
            cand = wrap_self(cur, w)
            if not is_palindrome(" ".join(cand)):
                continue
            sc = edge_score(cand, table, shapes)
            if sc >= base:
                key = (-sc, 0, len(w))
                if best is None or key < best[0]:
                    best = (key, cand, f"self:{w}", {w}, sc)
        # GLUE: multi-word wraps, attested ones first.
        for left, right, attested in glue:
            if used & (set(left) | set(right)):
                continue
            cand = wrap_glue(cur, left, right)
            if not is_palindrome(" ".join(cand)):
                continue
            sc = edge_score(cand, table, shapes)
            # An attested wrap beats an unattested one at equal edge score,
            # which is what the ordering in the key buys.
            if sc >= base:
                key = (-sc, 0 if attested else 1, len(left) + len(right))
                if best is None or key < best[0]:
                    best = (key, cand,
                            f"glue:{' '.join(left)}|{' '.join(right)}",
                            set(left) | set(right), sc)
        if best is None:
            break
        _, cur, op, words_used, base = best
        trail.append(op)
        used |= words_used
    return cur, trail


def seeds():
    """Every verified palindrome this project holds."""
    out = {}
    spelled = json.load(open("data/canon_spelled.json"))
    items = spelled.items() if isinstance(spelled, dict) else [(x, x) for x in spelled]
    for _, v in items:
        t = v if isinstance(v, str) else (v.get("text") if isinstance(v, dict) else str(v))
        if is_palindrome(t):
            out[" ".join(normalize(w) for w in t.lower().split() if normalize(w))] = "canon"
    for f in glob.glob("runs/polaris/*/v*.jsonl"):
        for line in open(f):
            t = json.loads(line)["text"]
            if is_palindrome(t):
                out[t] = "polaris"
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vocab", type=int, default=30000)
    ap.add_argument("--max-ops", type=int, default=6)
    ap.add_argument("--show", type=int, default=12)
    ap.add_argument("--out", default="experiments/extend_ops.json")
    args = ap.parse_args()

    vocab = build_vocab(args.vocab)
    selfpal, pairs = material(vocab)
    glue = glue_pairs()
    table, shapes, trig = brown_tables()
    src = seeds()
    print(f"{len(src)} verified seeds; {len(selfpal)} self-palindromic words, "
          f"{len(pairs)} reversible pairs, {len(glue)} multi-word glue wraps\n")

    rows = []
    for text, origin in src.items():
        w0 = text.split()
        grown, trail = grow(w0, selfpal, pairs, table, shapes, args.max_ops,
                            glue=glue)
        g = " ".join(grown)
        assert is_palindrome(g), g
        rows.append({"origin": origin, "seed": text,
                     "seed_letters": len(normalize(text)),
                     "grown": g, "grown_letters": len(normalize(g)),
                     "ops": trail})

    rows.sort(key=lambda r: -r["grown_letters"])
    print(f"{'seed':>5} {'grown':>6}  text")
    for r in rows[:args.show]:
        print(f"{r['seed_letters']:5d} {r['grown_letters']:6d}  "
              f"{present(r['grown'].split(), table, shapes, trig)}")
    with open(args.out, "w") as fh:
        json.dump(rows, fh, indent=2)
    gains = [r["grown_letters"] - r["seed_letters"] for r in rows]
    print(f"\nlongest {max(r['grown_letters'] for r in rows)} letters; "
          f"mean gain {sum(gains)/len(gains):.1f} letters; "
          f"{sum(1 for g in gains if g>0)}/{len(gains)} grew")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

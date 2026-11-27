"""Short palindromic sentences aimed at the middle of the readability spectrum.

Outside-in letter search (Norvig/Hoey) over Brown-attested words. Joins need
not be frequent web bigrams; a join is allowed when the Brown tag 4-grams
around it are attested, and words of middling rarity are paid for. Every
closure is then ranked by spectrum.Spectrum, so the output lands between
stock phrases and salad instead of at either end.

    python -m llm_palindrome.middle            # best per length band
"""
from __future__ import annotations

import argparse
import collections
import math

from wordfreq import zipf_frequency

from .generate import build_vocab
from .spectrum import Spectrum


class _Index:
    def __init__(self, words):
        self.set = set(words)
        self.rev = {w[::-1]: w for w in words}
        self.fp, self.rp = collections.defaultdict(list), collections.defaultdict(list)
        for w in words:
            for k in range(1, len(w)):
                self.fp[w[:k]].append(w)
                self.rp[w[::-1][:k]].append(w)

    def left(self, over):
        """(word, new_overhang, side owing) for a word appended to the left half."""
        out = [(over[:k], over[k:], "R") for k in range(1, len(over) + 1) if over[:k] in self.set]
        return out + [(w, w[len(over):], "L") for w in self.fp.get(over, ())]

    def right(self, over):
        """The same for a word prepended to the right half, matched reversed."""
        out = [(self.rev[over[:k]], over[k:], "L") for k in range(1, len(over) + 1) if over[:k] in self.rev]
        return out + [(w, w[::-1][len(over):], "R") for w in self.rp.get(over, ())]


def render(words) -> str:
    s = " ".join("I" if w == "i" else w for w in words)
    return s[0].upper() + s[1:] + "."


def search(sp: Spectrum, lo: int, hi: int, vocab_n=40000, beam=12000, max_words=12,
           per_start=80, max_over=9, unseen=-8.0, rare_w=0.6, zipf_mid=4.0) -> list[tuple]:
    t = sp.t
    words = [w for w in build_vocab(vocab_n) if t.is_word(w) and t.count[w] >= 4]
    idx = _Index(words)
    zb = {w: -rare_w * abs(zipf_frequency(w, "en") - zipf_mid) for w in words}
    fwd, tot = t.fwd, t.fwd_total
    lf = lambda a, b: math.log(fwd[a][b] / tot[a]) if b in fwd.get(a, ()) else unseen  # noqa: E731
    pad = ("<s>",) * 3

    def ok_left(lt, tag):
        return (pad + lt)[-3:] + (tag,) in t.tag4

    def ok_right(tag, rt):
        g = ((tag,) + rt + (".",))[:4]
        return len(g) < 4 or g in t.tag4

    states = [(lf("<s>", w) + zb[w], (w,), (), w, "L", (tag,), ())
              for w in words for tag in t.lex[w] if ok_left((), tag)]
    states = sorted(states, key=lambda s: -s[0])[:beam]
    closed = set()
    for _ in range(2 * max_words):
        pool = []
        for score, L, R, over, side, lt, rt in states:
            n = sum(map(len, L)) + sum(map(len, R))
            used = collections.Counter(L + R)
            if side == "R" or not over:
                if not over:
                    continue
                for w, nov, ns in idx.left(over):
                    if len(nov) <= max_over and n + len(w) <= hi:
                        sc = score + lf(L[-1], w) + zb[w] - 4.0 * used[w]
                        pool += [(sc, L + (w,), R, nov, ns, lt + (g,), rt)
                                 for g in t.lex[w] if ok_left(lt, g)]
            else:
                for w, nov, ns in idx.right(over):
                    if len(nov) <= max_over and n + len(w) <= hi:
                        sc = score + (lf(w, R[0]) if R else -6.0) + zb[w] - 4.0 * used[w]
                        pool += [(sc, L, (w,) + R, nov, ns, lt, (g,) + rt)
                                 for g in t.lex[w] if ok_right(g, rt)]
        if not pool:
            break
        for _, L, R, over, _, lt, rt in pool:
            if R and over == over[::-1] and sum(map(len, L + R)) >= lo:
                seq = pad + lt + rt + (".",)
                if all(seq[i:i + 4] in t.tag4 for i in range(len(seq) - 3)):
                    closed.add(L + R)
        pool.sort(key=lambda s: -s[0])
        per, seen, states = collections.Counter(), set(), []
        for st in pool:
            if st[1:5] in seen or per[st[1][0]] >= per_start:
                continue
            seen.add(st[1:5])
            per[st[1][0]] += 1
            states.append(st)
            if len(states) >= beam:
                break
    return sorted(((sp.assess(render(k)), render(k)) for k in closed),
                  key=lambda p: -p[0]["score"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bands", default="8-16,17-24,25-40")
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()
    sp = Spectrum()
    for band in args.bands.split(","):
        lo, hi = map(int, band.split("-"))
        ranked = search(sp, lo, hi)
        print(f"\n{lo}-{hi} letters: {len(ranked)} closed")
        for a, text in ranked[:args.top]:
            print(f"{a['score']:.2f}  pos {a['position']:.2f}  readable {a['readable']:.2f}  ism {a['ism']:.2f}  {text}")


if __name__ == "__main__":
    main()

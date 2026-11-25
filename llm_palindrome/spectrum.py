"""Readability spectrum: a fast, count-based verifier for short texts.

    garbage  <---------------  middle  --------------->  ism
    salad, filler,            grammatical, with          stock: every word
    repetition, non-words     content a reader           common, every join
                              would not predict          predictable, or a classic

No model call. Two axes, each from counts:

  readable  logistic fit (training/fit_spectrum.py) over Brown tag-trigram
            syntax gain (mean and worst step) and filler share, minus hard
            penalties for repeated content words and lowercase non-words.
  ism       web-bigram surprisal of the joins plus word rarity, mapped so the
            live v1 output and stock phrases sit at 1 and human r/palindromes
            titles sit near 0.2.

`score` rewards readable text at ISM_TARGET, so a search aims for the middle,
not either end. `position` places a text on the 0 (garbage) .. 1 (ism) line.
About 0.2 ms per text.
"""
from __future__ import annotations

import collections
import math
import os
import pickle
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIGRAMS = os.path.join(ROOT, "data", "count_2w.txt")  # norvig.com/ngrams/count_2w.txt
CACHE = os.path.join(ROOT, "data", ".spectrum_tables.pkl")

# The ism anchors. A text sharing most of its letters with one is a retread.
CLASSICS = [
    "a man a plan a canal panama", "step on no pets", "was it a car or a cat i saw",
    "never odd or even", "madam im adam", "too hot to hoot", "no lemon no melon",
    "do geese see god", "rats live on no evil star", "live not on evil",
    "a santa at nasa", "dammit im mad", "murder for a jar of red rum",
    "a nut for a jar of tuna", "borrow or rob", "draw o coward", "won ton not now",
    "yo banana boy", "taco cat", "senile felines", "no x in nixon",
    "go hang a salami im a lasagna hog", "able was i ere i saw elba", "evil olive",
    "stressed desserts", "top spot", "no devil lived on", "sit on a potato pan otis",
    "we panic in a pew", "red roses run no risk sir on nurses order",
    "pull up if i pull up", "now i won", "put it up",
    "no one made killer apparel like dame noon", "mr owl ate my metal worm",
    "ma has a ham", "nurses run", "drawn onward",
]

FUNCTION = set("""a an the i me my we us our you he him his she her it its they them their
is are was were be been am do does did has have had will would can could shall should may
might must of to in on at by for from with as or and but if so no not nor up out off
this that these those there here then than too very all any some each""".split())

# Fitted by training/fit_spectrum.py; rerun it after changing the features.
READABLE = {
    "keys": ["syntax", "worst", "filler"],
    "mu": [0.3147, -0.8202, 0.3860], "sd": [0.6121, 0.8119, 0.2282],
    "w": [0.9189, 0.6865, -0.2076], "b": 0.2976,
}


def _norm_tag(t: str) -> str:
    t = t.split("+")[0].split("-")[0]
    return t.rstrip("*$") or "X"


def words_of(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower().replace("'", "").replace("’", ""))


def clauses(text: str) -> list[list[str]]:
    """Clauses split at . ; : ? !, commas kept as tokens, and a capitalised
    word inside a clause marked '^' as a likely name."""
    out = []
    for part in re.split(r"[.;:?!]+", text.replace("'", "").replace("’", "")):
        toks: list[str] = []
        for m in re.finditer(r"[A-Za-z]+|,", part):
            w = m.group(0)
            if w == ",":
                if toks and toks[-1] != ",":
                    toks.append(",")
                continue
            name = w[0].isupper() and toks and w.lower() != "i"
            toks.append(("^" if name else "") + w.lower())
        while toks and toks[-1] == ",":
            toks.pop()
        if toks:
            out.append(toks)
    return out


class Tables:
    """Brown tag n-grams and lexicon plus web bigrams, built once and cached."""

    def __init__(self):
        from nltk.corpus import brown
        emit = collections.defaultdict(collections.Counter)
        tri, bi, uni, four = (collections.Counter() for _ in range(4))
        caps, seen = collections.Counter(), collections.Counter()

        def flush(tags):
            tags.append("</s>")
            if len(tags) <= 3:
                return
            for i in range(2, len(tags)):
                tri[tuple(tags[i - 2:i + 1])] += 1
                bi[tuple(tags[i - 1:i + 1])] += 1
                uni[tags[i]] += 1
            plain = ["<s>"] * 3 + [t for t in tags[2:-1] if t != ","] + ["."]
            for i in range(len(plain) - 3):
                four[tuple(plain[i:i + 4])] += 1

        for sent in brown.tagged_sents():
            tags = ["<s>", "<s>"]
            for w, t in sent:
                if w == ",":
                    if tags[-1] not in ("<s>", ","):
                        tags.append(",")
                    continue
                if w in (".", ";", ":", "?", "!"):
                    flush(tags)
                    tags = ["<s>", "<s>"]
                    continue
                if t.startswith("FW"):
                    tags.append("FW")  # foreign words never enter the lexicon
                    continue
                w = w.replace("'", "")
                if not w.isalpha():
                    continue
                if len(tags) > 2:
                    seen[w.lower()] += 1
                    caps[w.lower()] += w[0].isupper()
                t = _norm_tag(t)
                emit[t][w.lower()] += 1
                tags.append(t)
            flush(tags)

        self.uni, self.bi, self.tri = dict(uni), dict(bi), dict(tri)
        self.uni_total = sum(uni.values())
        self.tag4 = {g for g, n in four.items() if n >= 2}
        ctx2, ctx1 = collections.Counter(), collections.Counter()
        for (a, b, _), n in tri.items():
            ctx2[(a, b)] += n
        for (a, _), n in bi.items():
            ctx1[a] += n
        self.ctx2, self.ctx1 = dict(ctx2), dict(ctx1)

        wt = collections.defaultdict(dict)
        for t, c in emit.items():
            for w, n in c.items():
                wt[w][t] = n
        self.count = {w: sum(d.values()) for w, d in wt.items()}
        self.lex = {}  # word -> {tag: log P(tag | word)}
        for w, d in wt.items():
            keep = {t: n for t, n in d.items() if n / self.count[w] >= 0.02}
            s = sum(keep.values())
            self.lex[w] = {t: math.log(n / s) for t, n in keep.items()}
        # Mostly capitalised inside Brown sentences: names and acronyms. Written
        # in lowercase they read as non-words.
        self.names = {w for w, n in seen.items() if caps[w] / n > 0.5} - {"i"}

        fwd, tot = collections.defaultdict(dict), collections.Counter()
        with open(BIGRAMS, encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                try:
                    pair, n = line.rstrip("\n").split("\t")
                    a, b = pair.lower().split(" ")
                except ValueError:
                    continue
                if (a.isalpha() or a == "<s>") and b.isalpha():
                    fwd[a][b] = fwd[a].get(b, 0) + int(n)
                    tot[a] += int(n)
        self.fwd, self.fwd_total = dict(fwd), dict(tot)

    def is_word(self, w: str) -> bool:
        """Brown-attested, not foreign, not only ever a name."""
        tags = self.lex.get(w)
        return bool(tags) and not set(tags) <= {"NP", "NPS"} and w not in self.names

    @classmethod
    def load(cls) -> "Tables":
        if os.path.exists(CACHE):
            with open(CACHE, "rb") as fh:
                return pickle.load(fh)
        t = cls()
        with open(CACHE, "wb") as fh:
            pickle.dump(t, fh)
        return t


def _guess(w: str) -> dict[str, float]:
    for suf, t in (("ing", "VBG"), ("ed", "VBD"), ("ly", "RB"), ("ous", "JJ"),
                   ("ful", "JJ"), ("ive", "JJ"), ("al", "JJ"), ("s", "NNS")):
        if w.endswith(suf) and len(w) > len(suf) + 2:
            return {t: math.log(0.6), "NN": math.log(0.3), "NP": math.log(0.1)}
    return {"NN": math.log(0.5), "NP": math.log(0.35), "JJ": math.log(0.15)}


class Spectrum:
    SUR_ISM, SUR_EDGE = 4.5, 11.0  # surprise of stock text .. beyond human edgy
    ISM_TARGET, ISM_WIDTH = 0.2, 0.35
    R_LO, R_OK = 0.3, 0.6          # readable probability: garbage .. readable

    def __init__(self, tables: Tables | None = None):
        from wordfreq import zipf_frequency
        self.t = tables or Tables.load()
        self._zipf = zipf_frequency
        self._zc: dict[str, float] = {}

    def zipf(self, w: str) -> float:
        z = self._zc.get(w)
        if z is None:
            z = self._zc[w] = self._zipf(w, "en")
        return z

    # -- syntax ------------------------------------------------------------
    def _ptri(self, a, b, c) -> float:
        t = self.t
        p3 = t.tri.get((a, b, c), 0) / t.ctx2[(a, b)] if (a, b) in t.ctx2 else 0.0
        p2 = t.bi.get((b, c), 0) / t.ctx1[b] if b in t.ctx1 else 0.0
        return 0.6 * p3 + 0.3 * p2 + 0.1 * t.uni.get(c, 0) / t.uni_total + 1e-7

    def syntax(self, words: list[str]) -> list[float]:
        """Per-step gain along the best tag path: log P(tag | two previous)
        minus log P(tag), clause end included. Salad takes a step Brown never does."""
        t = self.t
        lu = lambda c: math.log(t.uni.get(c, 1) / t.uni_total)  # noqa: E731
        states = {("<s>", "<s>"): (0.0, ())}
        for w in words:
            if w == ",":
                cand = {",": 0.0}
            elif w.startswith("^"):
                cand = dict(t.lex.get(w[1:]) or {})
                cand["NP"] = max(cand.get("NP", -9.0), math.log(0.5))
            else:
                cand = t.lex.get(w) or _guess(w)
            nxt: dict = {}
            for (a, b), (s, g) in states.items():
                for c, lw in cand.items():
                    lt = math.log(self._ptri(a, b, c))
                    if s + lt + lw > nxt.get((b, c), (-1e18,))[0]:
                        nxt[(b, c)] = (s + lt + lw, g + (lt - lu(c),))
            states = dict(sorted(nxt.items(), key=lambda kv: -kv[1][0])[:12])
        end = lambda k: math.log(self._ptri(k[0], k[1], "</s>"))  # noqa: E731
        k = max(states, key=lambda k: states[k][0] + end(k))
        return list(states[k][1]) + [end(k) - lu("</s>")]

    # -- features ------------------------------------------------------------
    def surprise(self, words: list[str]) -> float:
        fwd, tot = self.t.fwd, self.t.fwd_total
        joins = [-math.log(fwd[a][b] / tot[a]) if b in fwd.get(a, ()) else 14.0
                 for a, b in zip(words, words[1:])]
        join = sum(joins) / len(joins) if joins else 14.0
        return 0.5 * join + sum(7.5 - self.zipf(w) for w in words) / max(1, len(words))

    @staticmethod
    def repetition(words: list[str]) -> float:
        c = collections.Counter(w for w in words if w not in FUNCTION)
        bis = list(zip(words, words[1:]))
        return (sum(n - 1 for n in c.values()) + 2 * (len(bis) - len(set(bis)))) / max(1, len(words))

    @staticmethod
    def classic(text: str) -> float:
        tape = re.sub("[^a-z]", "", text.lower())
        best = 0.0
        for c in CLASSICS:
            ct = c.replace(" ", "")
            if tape and (ct in tape or tape in ct):
                best = max(best, min(len(ct), len(tape)) / max(len(ct), len(tape)))
        return best

    def features(self, text: str) -> dict:
        w = words_of(text)
        cl = clauses(text)
        g = [x for c in cl for x in self.syntax(c)] or [-9.0]
        worst = sorted(g)[:2]
        toks = [x for c in cl for x in c if x != ","]
        unknown = [x for x in toks if not x.startswith("^") and not self.t.is_word(x)]
        return {"syntax": sum(g) / len(g), "worst": sum(worst) / len(worst),
                "filler": sum(x in FUNCTION or len(x) <= 2 for x in w) / max(1, len(w)),
                "surprise": self.surprise(w), "repetition": self.repetition(w),
                "oov": len(unknown) / max(1, len(toks)), "classic": self.classic(text)}

    # -- the spectrum ----------------------------------------------------------
    @staticmethod
    def readable(f: dict) -> float:
        W = READABLE
        z = W["b"] + sum(w * (f[k] - m) / s for k, w, m, s in zip(W["keys"], W["w"], W["mu"], W["sd"]))
        # Repeated content words and lowercase non-words beyond the share human
        # palindromes use are garbage by definition, not learned weights.
        return max(0.0, 1 / (1 + math.exp(-z)) - 2.0 * f["repetition"] - 1.5 * max(0.0, f["oov"] - 0.15))

    def assess(self, text: str) -> dict:
        f = self.features(text)
        r = self.readable(f)
        ism = (self.SUR_EDGE - f["surprise"]) / (self.SUR_EDGE - self.SUR_ISM)
        ism = min(1.0, max(0.0, ism, f["classic"]))
        ok = min(1.0, max(0.0, (r - self.R_LO) / (self.R_OK - self.R_LO)))
        stock = max(0.0, (ism - self.ISM_TARGET) / (1 - self.ISM_TARGET))
        score = ok * (0.5 + 0.5 * r) * math.exp(-((ism - self.ISM_TARGET) / self.ISM_WIDTH) ** 2)
        return {**{k: round(v, 3) for k, v in f.items()}, "readable": round(r, 3),
                "ism": round(ism, 3), "position": round(0.5 * ok * (1 + stock), 3),
                "score": round(score, 3)}

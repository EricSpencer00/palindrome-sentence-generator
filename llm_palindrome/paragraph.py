"""Long palindrome paragraphs, grown from the centre outward as nested shells.

    ... L3 L2 [p] L1 [centre] R1 [p] R2 R3 ...

R_i spells L_i backwards letter by letter, but word breaks need not line up:
a word on one side can straddle two on the other (the overhang carries across
shells). Wherever the overhang is empty, a palindrome p can go in on both
sides at once, since p reverses to itself: "racecar", "did", "noon", "level".

Punctuation is free, so sentences start and end wherever the prose wants, on
each side independently. A trigram model of literary prose scores every token
in its reading direction: the left half backward, since it grows by
prepending, the right half forward.

    python -m llm_palindrome.paragraph --letters 500
"""
from __future__ import annotations

import argparse
import collections
import functools
import heapq
import math
import random

from .contractions import SPELL
from .prose_lm import BOS, ProseLM
from .spectrum import FUNCTION, Tables
from .validator import is_palindrome

ENDS = (".", "?")  # tokens that close a sentence
MARKS = (",",) + ENDS  # free punctuation: no letters
PUNCT = ("",) + MARKS

# Palindromes that can sit at any mirrored seam, or at the centre.
PALINDROMES = """did eye noon level racecar madam refer civic deed peep pop sees stats solos
tenet gag nun kayak radar rotor redder wow mom dad pup tot sis ewe eve ere""".split()


def letters(tok: str) -> str:
    return "" if tok in MARKS else tok


class Lexicon:
    CAP = 150

    def __init__(self, lm: ProseLM, t: Tables, min_count: int = 3):
        from .shortwords import REAL_SHORT_WORDS
        # Stems the tokenizer cut from contractions: "didn" of "didn't".
        stubs = {"didn", "doesn", "don", "isn", "wasn", "weren", "hasn", "haven", "hadn",
                 "couldn", "wouldn", "shouldn", "mustn", "aren", "ain", "shan", "needn", "ll", "ve", "re"}
        self.words = {w for w, n in lm.uni.items()
                      if n >= min_count and w.isalpha() and w not in stubs
                      and (w in SPELL or t.is_word(w) and (len(w) > 2 or w in REAL_SHORT_WORDS))}
        self.freq = {w: lm.uni[w] for w in self.words}
        self.prefix, self.suffix = collections.defaultdict(list), collections.defaultdict(list)
        for w in self.words:
            for k in range(1, len(w)):
                self.prefix[w[:k]].append(w)
                self.suffix[w[-k:]].append(w)
        # Backward continuations for a free left step: words seen before b.
        self.before = collections.defaultdict(list)
        for (a, b), n in lm.bi.items():
            if a in self.words:
                self.before[b].append((n, a))
        for b in self.before:
            self.before[b] = [a for _, a in sorted(self.before[b], reverse=True)[:60]]

    def payable(self, owner: str, owed: str) -> bool:
        """Can the owing side make any progress on this debt at all?"""
        if not owed:
            return True
        if owner == "R":
            return bool(self.prefix.get(owed)) or any(owed[:k] in self.words for k in range(1, len(owed) + 1))
        return bool(self.suffix.get(owed)) or any(owed[-k:] in self.words for k in range(1, len(owed) + 1))

    @functools.lru_cache(maxsize=200000)
    def append_matches(self, owed: str):
        """Words the right half can append against `owed`: (word, rest, flipped).
        Overruns are capped to the most frequent CAP; every exact fit is kept."""
        out = [(owed[:k], owed[k:], False) for k in range(1, len(owed) + 1) if owed[:k] in self.words]
        over = sorted(self.prefix.get(owed, ()), key=lambda w: -self.freq[w])[:self.CAP]
        return tuple(out + [(w, w[len(owed):], True) for w in over])

    @functools.lru_cache(maxsize=200000)
    def prepend_matches(self, owed: str):
        """Words the left half can prepend against `owed`, filled from its end."""
        out = [(owed[-k:], owed[:-k], False) for k in range(1, len(owed) + 1) if owed[-k:] in self.words]
        over = sorted(self.suffix.get(owed, ()), key=lambda w: -self.freq[w])[:self.CAP]
        return tuple(out + [(w, w[:len(w) - len(owed)], True) for w in over])


class State:
    """Word uses and word pairs are materialised only for states that enter
    the beam; candidates carry a pointer to their parent and the delta."""
    __slots__ = ("seq", "owner", "owed", "score", "n", "inserts", "parent", "new_words",
                 "edges", "_uses", "_bigs", "adj")

    def __init__(self, seq, owner, owed, score, n, inserts, parent=None, new_words=(), edges=()):
        self.seq, self.owner, self.owed = seq, owner, owed
        self.score, self.n, self.inserts = score, n, inserts
        self.parent, self.new_words, self.edges = parent, new_words, edges
        self._uses = self._bigs = None
        self.adj = parent.adj if parent is not None else 0.0

    def settle(self):
        if self._uses is None:
            p = self.parent
            self._uses = p.uses.copy() if p else collections.Counter()
            self._uses.update(self.new_words)
            self._bigs = set(p.bigs) if p else set()
            self._bigs.update(self.edges)
            self.parent = None
        return self

    @property
    def uses(self):
        return self.settle()._uses

    @property
    def bigs(self):
        return self.settle()._bigs


def _content(w):
    return len(w) >= 3 and w not in FUNCTION


def _word_after(seq, i):
    return next((x for x in seq[i + 1:i + 4] if x not in MARKS), None)


def _word_before(seq, i):
    return next((x for x in reversed(seq[max(0, i - 3):i]) if x not in MARKS), None)


def _first_end(seq):
    return next(i for i, x in enumerate(seq) if x in ENDS)


def _last_end(seq):
    return max(i for i, x in enumerate(seq) if x in ENDS)


def grow(lm: ProseLM, lex: Lexicon, target: int, beam: int = 400, per_parent: int = 12,
         max_owed: int = 8, letter_w: float = 0.7, owed_w: float = 0.4, repeat_pen: float = 4.0,
         insert_pen: float = 1.0, window_pen: float = 4.0, top_k: int = 30, content_bonus: float = 0.8, cycle_pen: float = 6.0, seed: int = 0, centre: tuple[str, ...] = (), debug: bool = False,
         spectrum=None, sent_w: float = 1.5, comma_pen: float = 1.0,
         theme: dict | None = None, theme_w: float = 6.0,
         runon: int = 18, runon_pen: float = 2.0,
         neural=None, refresh: int = 3, neural_w: float = 1.0,
         recall: int = 40, far_pen: float = 0.5, slack: int = 80,
         start_noise: float = 0.3, start_steps: int = 8, on_step=None) -> list[State]:
    rng = random.Random(seed)
    pals = [p for p in PALINDROMES if p in lex.words and len(p) > 1]
    start = State(centre, "R", "", 0.0, sum(map(len, centre)), 0, None, tuple(centre)).settle()
    states, closed = [start], []
    memo: dict = {}
    topic = theme or {}
    theme_words = list(topic)[:400]
    judged: dict = {}

    def fterm(sq, i) -> float:
        return lm.fwd(sq[i - 2] if i >= 2 else None, sq[i - 1] if i >= 1 else None, sq[i])

    def sentence(toks) -> float:
        """A sentence just closed on one side, its end mark included: judge it
        whole with the Brown syntax verifier. Fragments and salad pay; real
        clauses earn."""
        toks = tuple(toks)
        if toks not in judged:
            if spectrum is None or not toks:
                judged[toks] = 0.0
            else:
                f = spectrum.features(render(toks))
                r = spectrum.readable(f)
                n = sum(1 for x in toks if x not in MARKS)
                judged[toks] = sent_w * (math.log(max(r, 0.02)) - math.log(0.5)) * min(n, 12) / 4
        return judged[toks]

    def priority(s):
        return s.score + s.adj + letter_w * s.n - owed_w * len(s.owed)

    def quality(k) -> float:
        """Mean per-token value of a closed paragraph, penalties included."""
        return (k.score + k.adj) / len(k.seq)

    def finish(sq) -> tuple[float, str]:
        """A closed paragraph: its first word starts a sentence, its last ends
        one, and its two outer sentences are judged whole like the rest.
        The last sentence closes on whichever end mark scores better; returns
        (delta, mark)."""
        d = lm.fwd(None, BOS, sq[0]) - fterm(sq, 0)
        if len(sq) > 1:
            d += lm.fwd(BOS, sq[0], sq[1]) - fterm(sq, 1)
        inner = any(x in ENDS for x in sq)
        if inner:
            d += sentence(sq[:_first_end(sq) + 1])
        a = sq[-2] if len(sq) > 1 else BOS
        last = sq[_last_end(sq) + 1:] if inner else ()

        def close(e):
            return lm.fwd(a, sq[-1], e) + (sentence(last + (e,)) if inner else 0.0)
        mark = max(ENDS, key=close)
        return d + close(mark), mark

    def tri_total(sq) -> float:
        return sum(fterm(sq, i) for i in range(len(sq)))

    def reanchor(group):
        """Replace each state's trigram estimate with the neural reading of
        the whole paragraph so far; children inherit the correction."""
        texts = [render(k.seq) for k in group]
        for k, nt in zip(group, neural.totals(texts)):
            k.adj = neural_w * nt - tri_total(k.seq)

    step = 0

    while states:
        pool = []
        for s in states:
            kids = []
            seq = s.seq

            def push(new_seq, owner, owed, delta, added, new_words, edges, ins=0):
                uses, bigs = s.uses, s.bigs
                pen = 0.0
                left_side = new_seq[:1] != seq[:1] or not seq
                near = [x for x in (new_seq[:9] if left_side else new_seq[-9:]) if x not in MARKS]
                recent = new_seq[:recall] if left_side else new_seq[-recall:]
                for i, w in enumerate(new_words):
                    n = uses[w] + 1 + new_words[:i].count(w)
                    if _content(w):
                        local = recent.count(w) - 1
                        # The theme pays for a word once; after that it is ordinary repetition.
                        pen += (repeat_pen * local + far_pen * max(0, n - 1 - local)
                                - content_bonus - (theme_w * topic.get(w, 0.0) if n == 1 else 0.0))
                    else:
                        pen += 0.6 * max(0, recent.count(w) - 3)
                    if near.count(w) > 1:
                        pen += window_pen
                # An edge that has gone too long without a full stop is a run-on.
                edge = new_seq[:runon] if left_side else new_seq[-runon:]
                if len(new_seq) > runon and not any(x in ENDS for x in edge):
                    pen += runon_pen * len(new_words)
                # A word pair that already occurs is the start of a cycle.
                edges = tuple(e for e in edges if None not in e)
                pen += cycle_pen * sum(e in bigs for e in edges)
                kids.append(State(new_seq, owner, owed, s.score + delta - pen,
                                  s.n + added, s.inserts + ins, s, tuple(new_words), edges))

            def ledge(nxt, w):
                return ((w, _word_after(nxt, 0)),)

            def redge(nxt, w):
                return ((_word_before(nxt, len(nxt) - 1), w),)

            def left_variants(w):
                """Prepend w, with optional punctuation after it."""
                for p in PUNCT:
                    toks = (w, p) if p else (w,)
                    if p and seq and seq[0] in MARKS:
                        continue
                    nxt = toks + seq
                    # One forward model for the whole paragraph: prepending
                    # changes the terms for the new tokens and the context of
                    # the two tokens after them. The leftmost token is scored
                    # without context until something is prepended to it.
                    k = len(toks)
                    d = sum(fterm(nxt, i) for i in range(min(len(nxt), k + 2)))
                    d -= sum(fterm(seq, i) for i in range(min(len(seq), 2)))
                    if p == ",":
                        d -= comma_pen
                    if p in ENDS and any(x in ENDS for x in seq):
                        d += sentence(seq[:_first_end(seq) + 1])
                    yield nxt, d

            def right_variants(w):
                for p in PUNCT:
                    toks = (p, w) if p else (w,)
                    if p and seq and seq[-1] in MARKS:
                        continue
                    nxt = seq + toks
                    d = 0.0
                    for i in range(len(seq), len(nxt)):
                        d += lm.fwd(nxt[i - 2] if i >= 2 else BOS, nxt[i - 1] if i >= 1 else BOS, nxt[i])
                    if p == ",":
                        d -= comma_pen
                    if p in ENDS and any(x in ENDS for x in seq):
                        d += sentence(seq[_last_end(seq) + 1:] + (p,))
                    yield nxt, d

            if not s.owed:
                # A shell boundary. Start a new shell on the left ...
                ctx = next((x for x in seq if x not in MARKS), BOS)
                opts = list(lex.before.get(ctx, ())[:40 if seq else 400])
                if topic:
                    opts += [w for w in theme_words if w in lex.words
                             and lm.bi.get((w, ctx))][:20]
                for w in dict.fromkeys(opts):
                    for nxt, d in left_variants(w):
                        push(nxt, "R", w[::-1], d, len(w), [w], ledge(nxt, w))
                # ... or set a palindrome into both sides of the seam.
                if seq:
                    for p in pals:
                        for nl, dl in left_variants(p):
                            seq_l = nl
                            d_r = lm.fwd(seq_l[-2] if len(seq_l) > 1 else BOS, seq_l[-1], p)
                            full = seq_l + (p,)
                            push(full, "R", "", dl + d_r - insert_pen, 2 * len(p), [],
                                 ledge(full, p) + redge(full, p), 1)
            elif s.owner == "R":
                a, b = (seq[-2] if len(seq) > 1 else BOS), (seq[-1] if seq else BOS)
                key = ("R", s.owed, a, b)
                cands = memo.get(key)
                if cands is None:
                    cands = lex.append_matches(s.owed)
                    if len(cands) > top_k:
                        cands = heapq.nlargest(top_k, cands, key=lambda c: max(
                            lm.fwd(a, b, c[0]), *(lm.fwd(b, e, c[0]) - 1.0 for e in ENDS)) + theme_w * topic.get(c[0], 0.0))
                    memo[key] = cands
                for w, rest, flipped in cands:
                    for nxt, d in right_variants(w):
                        push(nxt, "L" if flipped else "R", rest[::-1] if flipped else rest,
                             d, len(w), [w], redge(nxt, w))
            else:
                b, c2 = (seq[0] if seq else BOS), (seq[1] if len(seq) > 1 else None)
                key = ("L", s.owed, b, c2)
                cands = memo.get(key)
                if cands is None:
                    cands = lex.prepend_matches(s.owed)
                    if len(cands) > top_k:
                        cands = heapq.nlargest(top_k, cands, key=lambda c: max(
                            lm.bwd(c[0], b, c2), *(lm.bwd(c[0], e, b) - 1.0 for e in ENDS)) + theme_w * topic.get(c[0], 0.0))
                    memo[key] = cands
                for w, rest, flipped in cands:
                    for nxt, d in left_variants(w):
                        push(nxt, "R" if flipped else "L", rest[::-1] if flipped else rest,
                             d, len(w), [w], ledge(nxt, w))

            kids = [k for k in kids if len(k.owed) <= max_owed and lex.payable(k.owner, k.owed)]
            pool.extend(heapq.nlargest(per_parent if seq else len(kids), kids, key=priority))

        for k in pool:
            if not k.owed and k.n >= target and k.seq[0] not in MARKS and k.seq[-1] not in MARKS:
                d, mark = finish(k.seq)
                k.adj += d
                # The paragraph's own full stop is implicit; any other mark is printed.
                closed.append(k if mark == "." else State(k.seq + (mark,), k.owner, k.owed, k.score,
                                                           k.n, k.inserts, k))
        if len(closed) > 2000:
            closed = heapq.nlargest(500, closed, key=quality)
        # Diversity: one state per (outer context, owed); a little noise.
        best = {}
        for k in pool:
            key = (k.seq[:3], k.seq[-3:], k.owner, k.owed)
            pk = priority(k) + rng.random() * (start_noise if step < start_steps else 0.3)
            if key not in best or pk > best[key][0]:
                best[key] = (pk, k)
        states = [k for _, k in heapq.nlargest(beam, best.values(), key=lambda x: x[0])]
        states = [k.settle() for k in states if k.n < target + slack]
        step += 1
        if on_step is not None and states and step % 4 == 0:
            on_step(states[0])
        if neural is not None and states and step % refresh == 0:
            reanchor(states)
            states.sort(key=priority, reverse=True)
        if debug and states:
            top = states[0]
            print(len(pool), len(states), len(closed), top.n, top.owner, repr(top.owed), render(top.seq), flush=True)
    return sorted(closed, key=quality, reverse=True)


def render(seq) -> str:
    out, cap = [], True
    for tok in seq:
        if tok in MARKS:
            if out:
                out[-1] += tok
            cap = tok in ENDS
            continue
        w = "I" if tok == "i" else SPELL.get(tok, tok)
        if cap:  # "'tis" opens a sentence as "'Tis"
            k = 1 if w[0] == "'" else 0
            w = w[:k] + w[k].upper() + w[k + 1:]
        out.append(w)
        cap = False
    text = " ".join(out)
    return text if text.endswith(ENDS) else text.rstrip(",") + "."


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--letters", type=int, default=400)
    ap.add_argument("--beam", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--centre", default="")
    ap.add_argument("--show", type=int, default=3)
    ap.add_argument("--theme", default="", help="comma-separated words, e.g. sea,ship,storm")
    args = ap.parse_args()
    lm, t = ProseLM.load(), Tables.load()
    lex = Lexicon(lm, t)
    from .prose_lm import theme_bonus, theme_vectors
    from .spectrum import Spectrum
    theme = theme_bonus(theme_vectors(), args.theme.split(",")) if args.theme else None
    closed = grow(lm, lex, args.letters, beam=args.beam, seed=args.seed, spectrum=Spectrum(t), theme=theme,
                  sent_w=2.0, content_bonus=1.6,
                  centre=tuple(args.centre.lower().split()))
    ranked = closed
    seen = set()
    for s in ranked:
        text = render(s.seq)
        if text in seen:
            continue
        seen.add(text)
        assert is_palindrome(text), text
        print(f"\n[{s.n} letters, {sum(1 for x in s.seq if x not in MARKS)} words, "
              f"lm {lm.score(list(s.seq)):.2f}, inserts {s.inserts}]\n{text}")
        if len(seen) >= args.show:
            break


if __name__ == "__main__":
    main()

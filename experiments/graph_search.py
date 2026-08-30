"""Search the overhang graph once instead of redrawing from it forever.

The overhang is a sufficient statistic: every partial palindrome with overhang
`o` owed by side `s` has exactly the same legal continuations. The existing
enumerator ignores this and re-walks shared subtrees on every branch, which is
why 2M draws at 32-36 letters on Polaris rediscovered one core and nothing
else.

Here the walk is a graph problem:

  1. build the reachable (overhang, owner) graph once -- every overhang is a
     suffix of some word's letters or reversed letters, so the graph is small
     even when the number of walks through it is astronomical;
  2. count closures by dynamic programming over (state, remaining letters),
     acyclic because letters strictly increase along every edge;
  3. sample closures uniformly at an exact letter count by walking down the
     count table. Every sample is a valid in-band palindrome by construction;
     the redraw walk spends nearly all of its budget on branches that never
     close in band.

Counts are float64: exact until they pass 2**53, and beyond that they are only
ever used as sampling weights, where relative error of 1e-16 is irrelevant.

Uniform sampling over closures fails outright -- the 32-36 band at 6,000 words
holds 1.6e14 closures and 25,204 uniform samples contained zero readable ones,
because uniformity spreads the draw across texts of uniformly rare words. The
redraw walk was findable-by-accident: its candidate ranking gave it a frequency
bias. `--alpha` makes that bias explicit and exact: each edge carries weight
freq(word)^alpha, so a sample is drawn with probability proportional to the
product of its words' frequencies. alpha=0 is uniform; alpha grows the bias.

The hit filter -- 3 to 9 units, `sentence_like` on the same Brown payload --
is copied from tools/polaris/shard_yield.py so rates are comparable with the
two Polaris jobs. Two of its rejections are structural and are pushed into the
graph rather than paid at sampling time: a word absent from the Brown table
fails `sentence_like` unconditionally, so the graph is built over tagged words
only; and the unit-count window is a DP dimension, so every sample has 3 to 9
units by construction instead of most samples being auto-rejects.
"""
from __future__ import annotations

import argparse, importlib.util, json, random, sys, time

sys.path.insert(0, ".")
import numpy as np

from llm_palindrome.search import WordTries, consume, unit_letters
from llm_palindrome.validator import is_palindrome, normalize

_spec = importlib.util.spec_from_file_location(
    "shard_yield", "tools/polaris/shard_yield.py")
_sy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sy)


def expand(overhang: str, owner: str, tries: WordTries, limit: int = 10 ** 6):
    """(word, placement, new_overhang, new_owner): search._expand, text-free."""
    out = []
    if owner == "L" or not overhang:
        for w in tries.right_candidates(overhang, limit):
            res = consume(unit_letters(w)[::-1], overhang)
            if res is not None:
                new_over, flipped = res
                out.append((w, "R", new_over, "R" if flipped else "L"))
    if owner == "R" and overhang:
        for w in tries.left_candidates(overhang, limit):
            res = consume(unit_letters(w), overhang)
            if res is not None:
                new_over, flipped = res
                out.append((w, "L", new_over, "L" if flipped else "R"))
    return out


class Graph:
    def __init__(self, tries: WordTries, max_overhang: int, alpha: float = 0.0):
        from wordfreq import zipf_frequency
        self.alpha = alpha
        # zipf is log10 of frequency per billion; freq^alpha == 10**(zipf*alpha)
        # up to a constant that cancels in sampling. Offset keeps weights
        # in a sane float range.
        self.wt = {}
        self.tries = tries
        t0 = time.time()
        self.keys: list[tuple[str, str]] = [("", "R")]
        self.ids: dict[tuple[str, str], int] = {("", "R"): 0}
        self.edges: list[list[tuple[str, str, int, int]]] = [[]]
        frontier = [0]
        while frontier:
            i = frontier.pop()
            o, s = self.keys[i]
            es = []
            for w, placement, new_over, new_owner in expand(o, s, tries):
                if len(new_over) > max_overhang:
                    continue
                nk = (new_over, new_owner)
                j = self.ids.get(nk)
                if j is None:
                    j = self.ids[nk] = len(self.keys)
                    self.keys.append(nk)
                    self.edges.append([])
                    frontier.append(j)
                if w not in self.wt:
                    z = zipf_frequency(w, "en")
                    self.wt[w] = 10.0 ** (alpha * (z - 4.0)) if z else 1e-6
                es.append((w, placement, len(unit_letters(w)), j))
            self.edges[i] = es
        self.n = len(self.keys)
        self.n_edges = sum(len(e) for e in self.edges)
        self.build_seconds = round(time.time() - t0, 1)

    def closure_counts(self, max_letters: int, max_units: int = 9) -> np.ndarray:
        """count[i, r, u]: unit sequences from state i closing in exactly r
        more letters using exactly u more units. Vectorised as one scatter-add
        per (word length, r, u)."""
        t0 = time.time()
        src, dst, dlt, ewt = [], [], [], []
        for i, es in enumerate(self.edges):
            for w, _, d, j in es:
                src.append(i); dst.append(j); dlt.append(d)
                ewt.append(self.wt[w])
        src = np.array(src); dst = np.array(dst); dlt = np.array(dlt)
        ewt = np.array(ewt)
        count = np.zeros((self.n, max_letters + 1, max_units + 1))
        for i, (o, _) in enumerate(self.keys):
            if o == "":
                count[i, 0, 0] = 1.0
        for r in range(1, max_letters + 1):
            for u in range(1, max_units + 1):
                for d in np.unique(dlt):
                    if d > r:
                        continue
                    m = dlt == d
                    np.add.at(count[:, r, u], src[m],
                              ewt[m] * count[dst[m], r - d, u - 1])
        self.count_seconds = round(time.time() - t0, 1)
        return count

    def sample(self, count: np.ndarray, letters: int, units: int,
               rng: random.Random):
        """One draw over closures with exactly `letters` letters, `units` units,
        weighted by the product of the words' frequency weights."""
        i, r, u = 0, letters, units
        left: list[str] = []
        right: list[str] = []
        while True:
            o = self.keys[i][0]
            weights = []
            opts = []
            if o == "" and r == 0 and u == 0:
                weights.append(1.0); opts.append(None)
            for w, placement, d, j in self.edges[i]:
                if d <= r and u >= 1 and count[j, r - d, u - 1] > 0:
                    weights.append(self.wt[w] * count[j, r - d, u - 1])
                    opts.append((w, placement, d, j))
            if not opts:
                return None
            pick = rng.choices(opts, weights=weights)[0]
            if pick is None:
                return left + right
            w, placement, d, j = pick
            if placement == "L":
                left.append(w)      # grown by appending toward the centre
            else:
                right.insert(0, w)  # right half in final order
            i, r, u = j, r - d, u - 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vocab", type=int, default=6000)
    ap.add_argument("--lo", type=int, default=32)
    ap.add_argument("--hi", type=int, default=36)
    ap.add_argument("--max-overhang", type=int, default=12)
    ap.add_argument("--seconds", type=float, default=300.0)
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="frequency bias; 0 is uniform over closures")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/graph_search.jsonl")
    args = ap.parse_args()

    words = open("tools/polaris/payload/vocab30k.txt").read().split()[:args.vocab]
    table, shapes, _ = _sy.load_brown("tools/polaris/payload/brown.json.gz")
    tagged = [w for w in words if w in table]
    print(f"vocab {len(words)} -> {len(tagged)} with Brown tags", flush=True)
    tries = WordTries(tagged)

    g = Graph(tries, args.max_overhang, alpha=args.alpha)
    print(f"graph: {g.n:,} states, {g.n_edges:,} edges, "
          f"built in {g.build_seconds}s", flush=True)
    count = g.closure_counts(args.hi)
    cells = {(r, u): count[0, r, u]
             for r in range(args.lo, args.hi + 1)
             for u in range(3, count.shape[2])
             if count[0, r, u] > 0}
    total = sum(cells.values())
    print(f"counts in {g.count_seconds}s; weighted closures in band, "
          f"3-9 units: {total:.3e} over {len(cells)} (letters, units) cells",
          flush=True)

    rng = random.Random(args.seed)
    lens, wts = list(cells), [cells[k] for k in cells]
    seen, hits = set(), []
    n_samples = 0
    t0 = time.time()
    deadline = t0 + args.seconds
    while time.time() < deadline:
        r, u = rng.choices(lens, weights=wts)[0]
        units = g.sample(count, r, u, rng)
        n_samples += 1
        if units is None:
            continue
        key = normalize(" ".join(units))
        if key in seen:
            continue
        seen.add(key)
        if _sy.sentence_like(units, table, shapes):
            text = " ".join(units)
            assert is_palindrome(text), text
            hits.append(text)
            print(f"  HIT {text}", flush=True)
    dt = time.time() - t0
    out = {"vocab": args.vocab, "alpha": args.alpha,
           "lo": args.lo, "hi": args.hi,
           "graph_states": g.n, "graph_edges": g.n_edges,
           "build_seconds": g.build_seconds, "count_seconds": g.count_seconds,
           "sample_seconds": round(dt, 1), "samples": n_samples,
           "distinct": len(seen), "hits": len(hits), "texts": hits}
    with open(args.out, "a") as fh:
        fh.write(json.dumps(out) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "texts"}), flush=True)


if __name__ == "__main__":
    main()

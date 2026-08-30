"""Does coherence degrade per seam, or per letter?

Nesting mirror-pairs makes length free. `non academia` + `reno sir parasites`
+ `set i sara prisoner` + `aimed a canon` is 54 letters and a valid
palindrome — longer than the 51-letter human best — and it reads worse than
either palindrome it was built from.

Two explanations, and they imply opposite strategies:

  per-seam    every join between two unrelated pairs costs something, so the
              route to a long readable palindrome is FEW, LONG chunks and the
              search should hunt for bigger single finds
  per-letter  length itself is what costs, so chunk count is irrelevant and
              nothing about assembly can help

The test holds the material constant and varies only the number of pairs
nested. If preference against the single seed collapses between k=1 and k=2 it
is per-seam; if it degrades smoothly with the letter count it is per-letter.
"""
from __future__ import annotations

import json, random, sys
sys.path.insert(0, '.')
from llm_palindrome.present import present
from llm_palindrome.syntax import brown_tables
from llm_palindrome.validator import is_palindrome, normalize

sys.path.insert(0, 'server')
from server.v3 import harvest_pair

def nest(pairs):
    """L1..Lk Rk..R1 — no centre needed when every unit is a mirror-pair."""
    words = []
    for l, _ in pairs: words += l
    for _, r in reversed(pairs): words += r
    return words

def build(bank, k, rng):
    pool = []
    for row in bank:
        g = harvest_pair(row["text"].split())
        if g and normalize(" ".join(g[0])) != normalize(" ".join(g[1])):
            pool.append(g)
    rng.shuffle(pool)
    return nest(pool[:k])

if __name__ == "__main__":
    bank = json.load(open("data/v3_bank.json"))
    t_, s_, g_ = brown_tables()
    rng = random.Random(0)
    print(f"{'pairs':>5} {'letters':>8}  text")
    for k in (1, 2, 3, 5, 8):
        w = build(bank, k, random.Random(k))
        assert is_palindrome(" ".join(w))
        print(f"{k:5d} {len(normalize(' '.join(w))):8d}  "
              f"{present(w, t_, s_, g_)[:150]}")

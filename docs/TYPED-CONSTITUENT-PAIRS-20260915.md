# Typed constituent reverse-tape pairs (15 September 2026)

This bounded experiment tests the simplest omitted operator: independently
construct complete ordinary multiword constituents on both sides, then pair
only when their normalized letter tapes are exact reverses. The right side is
not produced by splitting the left tape; therefore word-boundary crossings are
allowed. Templates include determiner NPs, adjective NPs, transitive clauses,
copular adjective clauses, and prepositional constituents/adjuncts.

Replay:

```bash
python3 experiments/typed_constituent_pair_search_20260915.py \
  --out runs/typed_constituent_pair_search_20260915/result.json \
  --max-phrases 300000 --seed 7
```

The run generated 289,522 typed phrases and indexed their normalized tapes.
Every returned pair was checked by an independent `normalize(text) ==
normalize(text)[::-1]` validator. A hard anti-shortcut gate required at least
six words total, no more than two words of length <=2, no repeated words,
non-identical sides, and at least one crossed word boundary. It returned **zero
exact pairs**, hence no readable hit (and no human-readability claim).

The mechanical checksum and manifest are preserved in
`runs/typed_constituent_pair_search_20260915/result.json`. The next concrete
operator is typed residual attachment: retain a typed constituent pair's
unmatched boundary remainder and search a third, independently typed PP/NP
attachment against that remainder, with the same anti-shortcut gate.

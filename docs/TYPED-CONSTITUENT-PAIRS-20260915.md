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
`runs/typed_constituent_pair_search_20260915/result.json`. A follow-up probe
adds a variable-length clause lattice: transitive and copular clauses receive
an optional common adverbial modifier such as `today`, `quietly`, or `outside`
as a typed edge. Both sides remain independently generated and are matched
only by reversed normalized tapes. The expanded replay generated 966,363
phrases but returned zero exact pairs under the same gates. This is concrete
negative evidence; the next operator is typed residual attachment, retaining
a boundary remainder and searching a third independently typed PP/NP
attachment against it.

# Fresh boundary-ending replacement (19 September 2026)

This small operator holds a fresh opening and middle constituent fixed, then
replaces only the ending constituent. It requires a live outside-in match and
logs the 4/8/12/16-character gates plus the first mismatch. It does not sweep
the inventory or reverse a completed sentence.

Replay:

```bash
python3 experiments/fresh_boundary_ending_replacement_20260919.py
```

The run has 12 rows, zero exact closures, and independent two-pointer plus
SHA-256 forward/reverse tape audits. The strongest row is rendered exactly as:

> I saw the careful keeper mark the ledger, was I.

It reaches 4 outer characters, but not 8, 12, or 16; its first continuation
mismatch is recorded in the JSON artifact. It is retained as a structural
near-miss, not a readability claim. The reader gate remains closed.

The concrete next repair is to replace only that row's first mismatching ending
constituent while preserving its opening and middle, then rerun the same four
reach gates.

# Polar-boundary slot repair (2026-09-18)

This bounded lane varied typed noun, adjunct, subject, and answer slots around
the longer polar-question boundary. It explicitly rejects direct reversible
word pairs and every proper multiword palindrome before mechanical admission.
It is distinct from the 710-clause sweep.

Run: `runs/polar-boundary-slot-repair-20260918-v2.json`

- 240 authored polar frames; 4,800 pair worlds checked (the complete bounded
  bank for this lane)
- 4 exact strings, all rejected because they contain hidden palindromic spans
- 0 mechanically admitted and 0 reader-eligible candidates

Representative prose-shaped near-miss:

> Was Noel an era?, saga, Leon saw.

It has 3 character mismatches. The closest exact rejected construction was:

> Was Noel a gas?, saga, Leon saw.

but `a gas saga` is a hidden palindromic span, so it is not a valid result.
The 44-letter diagnostic remains excluded for the same reason.

All rows carry independent two-pointer and forward/reverse SHA-256 audits,
slot provenance, novelty status, and reader status. Programmatic checks are
diagnostic only. Next repair: solve the noun/adjunct seam residual with
held-out verbs while forbidding reversible words and hidden spans.

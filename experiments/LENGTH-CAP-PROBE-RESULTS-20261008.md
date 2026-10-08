# Exploratory length-cap probe — 2026-10-08

Sixteen frozen jobs used two seeds, two native search implementations, two
letter bands and word caps 32/256. All other settings and vocabulary were
shared. This is development evidence, separate from the 360-job paper study.

| Letter band | Word cap | Exact outputs / jobs | Strict passes / jobs | Deadline exits |
|---|---:|---:|---:|---:|
| 120–159 | 32 | 3/4 | 1/4 | 0 |
| 120–159 | 256 | 4/4 | 2/4 | 0 |
| 800–999 | 32 | 0/4 | 0/4 | 0 |
| 800–999 | 256 | 2/4 | 0/4 | 2 |

The vocabulary's longest word is 18 letters. Therefore cap32 has an absolute
576-letter upper bound and cannot reach the 800-letter band. This bound was
noticed after execution. Its zero yield is a feasibility control, not evidence
that a tuned method beats a valid baseline. Two seeds cannot support a robust
performance estimate.

Both long outputs have 804 normalized letters and 184 words. They share a
word multiset and come from seed9000, so are related constructions rather than
independent linguistic successes. They independently reverse exactly but fail
`distinct_words` and `no_self_palindromic_proper_multiword_span`. They are not
readable passages and have no human or model readability ratings. Full texts
are in `runs/length-cap-probe-20261008/long-exact-804.txt`.

The first SSH session ended after four checkpointed results. No live runner
remained. Twelve uncheckpointed jobs were restarted; their original outcomes
and compute consumption are unknown. The initial checkpoint and continuation
record are retained. The final inventory has all16 specified jobs, no recorded
job errors. The 60-second budget is cooperative; deadline flags describe the
recorded executions, not unobserved attempts.

Offline verification: `python3 experiments/verify_length_cap_probe_20261008.py`.
It checks protocol/result joins, vocabulary and text hashes, full reversal,
length, word cap, strict flags and the interrupted checkpoint. It requires
neither inference nor a corpus download.

A separate local verification reproduced the complete mechanical check map
for all9 selected outputs against the frozen admission implementation.

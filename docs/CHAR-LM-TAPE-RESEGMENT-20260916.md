# Character-LM constrained decoding and tape resegmentation (16 September 2026)

The preflight read all 234 registry entries and rejected duplicate independent
seam/word sweeps and post-hoc tape reversal. This lane is one new operator:
score ordinary seed prose with a transparent character-trigram model, then
solve the reverse letter tape with dynamic-programming word boundaries.

The run produced five ordinary-English left proposals and one exact closure:
“a man a plan a canal panama” resegmented to the same words. The other four
remain useful non-exact prose candidates but have no complete reverse
segmentation under the bounded vocabulary. Every row records both tapes and
is replayable from the generator and seed hashes in the run artifact.

Run: `runs/char-lm-tape-resegment-20260916.json`. Exactness is a hard tape
equality check, not a proxy metric. The concrete next repair is a held-out
corpus character LM with the same DP boundary constraints, followed by human
readability review.

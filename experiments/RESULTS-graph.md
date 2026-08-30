# The overhang graph, searched once, against the walk that redraws

Run 29 August 2026. `experiments/graph_search.py`.

## The change

The overhang is a sufficient statistic, and the repository proved it and then
did not use it: `enumerate_palindromes` re-walks shared subtrees on every
branch, which is why 2M draws at 32-36 letters across two Polaris jobs
rediscovered one core and found nothing else.

`graph_search.py` builds the reachable (overhang, owner) graph once — every
overhang is a suffix of a word's letters or reversed letters, so the 6,000-word
graph is 22k states and 43k edges and builds in 0.1 seconds — then counts
closures exactly by dynamic programming over (state, letters, units), and
samples closures from that table. Every sample is a valid palindrome in the
target band with 3-9 units over Brown-tagged words, by construction.

Three design decisions, each forced by a measured failure:

1. **Uniform over closures fails.** The 32-36 band holds 1.6e14 closures and
   25,204 uniform samples contained zero readable ones: uniformity spreads the
   draw over texts of uniformly rare words. Edges carry weight
   freq(word)^alpha instead (alpha=0.3 here), so a draw's probability is
   proportional to the product of its words' frequencies — the bias the walk
   had by accident, made explicit and exact.
2. **Structural rejections belong inside the DP.** A word without a Brown tag
   fails `sentence_like` unconditionally, and so does any text outside 3-9
   units. Filtering afterwards threw away nearly every sample; building the
   graph over tagged words and adding a units dimension took the small test
   cell from 0 hits to 230 in the same fifteen seconds.
3. **The hit filter is imported from `tools/polaris/shard_yield.py`**, not
   reimplemented, so rates are comparable with both Polaris jobs.

## Head to head

Same vocabulary file, same filter, same core clustering (middle fourteen
letters). One laptop core against Polaris core-seconds.

| cell | method | throughput | hits | cores | rate |
|---|---|---|---:|---:|---|
| v6000 27-31 | walk, job 7553051 | 51,220 core-s | 212 | 12 | 2.3e-4 cores/core-s |
| v6000 27-31 | graph | 240 s | 351 | 330 | **1.38 cores/s** |
| v28402 32-36 | walk, two jobs | 2.07M draws | 2 | 1 | ~1.7e-5 cores/core-s |
| v28402 32-36 | graph | 900 s | 242 | 236 | **0.26 cores/s** |

At 27-31 letters the graph finds cores about **6,000x** faster per core-second.
At 32-36 — the band where two Polaris jobs together found one core — it is
about **15,000x** faster, and returned 236 distinct cores in fifteen minutes:

    no one sir nodes are erased on rise noon
    cain am animals was saw slam in a maniac
    a resist rats sup mac campus start sis era
    camera day baseman names a by a dare mac

## The bands nothing had ever reached

Same fifteen-minute budget per cell, full vocabulary:

| band | walk, ever | graph, 900 s | hits/sample |
|---|---|---|---|
| 32-36 | 1 core in 2.07M draws | 236 cores | 6.9e-3 |
| 37-41 | **0** in 189,201 draws | **141 cores** | 5.0e-3 |
| 42-51 | never attempted | **80 cores** | 3.1e-3 |

    [40] pets trade disputes tub but setup sided art step
    [40] dairy mar at noon drawer reward noon tara myriad
    [46] levers draw at eliot drawer reward toilet awards revel
    [46] name tart in level desserts stressed level nitrate man

Hits per sample fall barely — about 2.2x across fifteen letters — where the
walk's rates fell geometrically. The measured "decay" in
`RESULTS-polaris-yield.md` was to a large degree a property of the walk's
sampling, not of the space. That claim needs its own write-up with the
proposal distribution held fixed, because hits-per-sample under alpha=0.3 and
cores-per-draw under the walk are different measures; what is already safe to
say is that the 37+ bands are populated, densely, and were never empty.

The caveat that stops this being a readability claim: `sentence_like` is a
generous filter, and the long finds lean on formulaic mirror-pairs
(`desserts`/`stressed`, `drawer`/`reward`, `level`). The gated judge
(gpt-oss:20b, 0-3 scale) scored all 256 of them: **225 at 0, 31 at 1, none
higher.** On the same instrument our short chunks average 0.65, the catalogued
human palindromes 1.0, and the hand-punctuated 51-letter record 2 to 3. Supply
at 37-51 letters is solved; coherence there is exactly as absent as the seam
result predicts. The ceiling was never the search.

## What this changes

- The decay measurement (`RESULTS-polaris-yield.md`, pooled 2.08x per letter,
  CI [1.32, 3.71]) was rate-limited by the walk, not by the space. The graph
  can now put real counts in every band, including 37+ where nothing has ever
  been measured.
- Polaris debug jobs for yield are obsolete at these lengths. One laptop core
  outproduces a 32-rank node by orders of magnitude.
- `sentence_like` is now the binding filter: the search produces in-band
  material faster than any judge can read it, so the quality bar, not the
  supply, is the constraint from here.

## What it does not change

The seam result and the coherence ceiling are untouched — this is a faster way
to mine chunks, not a way to make them cohere. The samples above read like the
walk's samples, because they are drawn from the same population under the same
filter. And alpha=0.3 was chosen by a two-point sweep at a small cell, not
tuned; the throughput numbers move with it.

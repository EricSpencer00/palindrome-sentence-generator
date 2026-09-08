# Vocabulary scaling after the yield run

Job `7590268`, Polaris debug queue, 3--4 September 2026. One node, 32 MPI
ranks, `tools/polaris/scaling.pbs`; successful exit (`0`) after 41m51s. Raw
per-rank artifacts are preserved in `runs/polaris/scale_20260904_030603/`.

## Question

The earlier three-point Polaris sweep left an apparent tension: larger
vocabularies could express more material, but every additional word widened the
walk. This job filled in the vocabulary ladder at 27--31 letters and gave the
6,000-word walk enough time in the next two bands to measure its tail.

## Results

A **core** is the middle fourteen normalized letters, so rewordings of the
same palindrome do not inflate the finding count. `hits` are only the
permissive Brown-tag `sentence_like` filter, not a reading or coherence claim.

| vocab | band | distinct | hits | cores | cores / 1k core-s |
|---:|:---:|---:|---:|---:|---:|
| 600 | 27--31 | 1,576,762 | 18 | 8 | 1.387 |
| 2,400 | 27--31 | 439,337 | 92 | **12** | **2.079** |
| 4,800 | 27--31 | 227,068 | 109 | 9 | 1.557 |
| 9,600 | 27--31 | 111,083 | 89 | 6 | 1.034 |
| 19,200 | 27--31 | 52,555 | 73 | 1 | 0.171 |
| 6,000 | 32--36 | 3,064,137 | 25 | 3 | 0.104 |
| 6,000 | 37--41 | 626,745 | 0 | 0 | 0.000 |

Raw hit rate rises with vocabulary, but independent-core yield peaks at 2,400
in this sweep and then falls. The wide arms are increasingly rewordings of a
small number of centres, not a broader supply of meaningful finds. The
32--36-cell's three cores are variants of `draw delivered der evil edward`;
they are evidence of structural reach, not readable sentences.

## Decision

Do not run more Polaris yield walks. The direct graph sampler in
`experiments/graph_search.py` already outperforms the earlier walk by orders of
magnitude and reaches 37--51 letters locally (`RESULTS-graph.md`). This job is
still useful as the final control showing why the redraw walk is the wrong
instrument: it loses throughput as vocabulary grows while repeatedly finding
the same structural families.

## Limits

One seed per vocabulary cell and a heuristic core definition mean this is not
a claim that 2,400 is a universal optimum. It is enough to rule out a larger
CPU walk as the next research investment. The graph result independently makes
the same operational decision.

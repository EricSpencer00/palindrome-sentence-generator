# Hard-diversity search debug screen

Date: 2026-09-04  
Polaris job: `7592453.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov`  
Queue: debug, one Polaris node, 32 ranks  
Wall time: 00:01:34  
Exit status: 0  
Artifacts: `runs/polaris/diversity_debug_20260904_194448/`

## Question

Does making diversity a search constraint prevent the semantic scorer's
single-template collapse, and does the surviving low-weight semantic signal
improve word order under a held-out measurement?

The constrained arms partitioned 1,024 possible opening words disjointly over
32 ranks, removed an opening after using it on a rank, and capped every word at
two uses. The semantic arm added the previously screened order-gain term at
weight 0.25. All other search settings and seeds were matched.

## Search results

| arm | closed | texts | openings | length templates | distinct-word ratio | attested pairs |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 128/128 | 15 | 5 | 8 | 66.1% | 75.8% |
| hard diversity | 59/128 | 59 | 59 | 57 | 79.1% | 68.3% |
| hard diversity + semantic proposal | 59/128 | 59 | 59 | 57 | 77.3% | 81.4% |

All 246 closed outputs were exact validated palindromes. The constrained arms
gave up closure rate (46.1%) but eliminated duplicate outputs and opening
collapse. The semantic proposal recovered 13.1 points of attested adjacency
without reducing that coverage.

## Held-out order evaluation

Every distinct output was scored by GPT-2 per predicted token, then compared
with three deterministic shuffles of its own words. The difference measures
what order earned while holding the vocabulary fixed.

| arm | mean raw GPT-2/token | mean gain over own shuffles | best gain | Pareto survivors* |
|---|---:|---:|---:|---:|
| baseline | -5.803 | -0.091 | +0.090 | 0 |
| hard diversity | -6.122 | +0.066 | +0.515 | 1 |
| hard diversity + semantic proposal | -5.905 | **+0.229** | **+0.646** | **9** |

\* Non-dominated on shuffle gain, distinct-word ratio, and attested-pair rate.

## Decision

Promote hard opening partitions, the two-use cap, and order weight 0.25 as the
new research search baseline. This combination passes the stated gate: it
improves held-out order score, expands unique openings rather than reducing
them, and produces no repeated text or length-template collapse.

This does **not** pass the prose-quality north star. The strongest candidates
still read as locally improved word salad. The result establishes a healthier
candidate distribution for the next composer or judge; it is not evidence
that local adjacency is sufficient for sentence meaning.

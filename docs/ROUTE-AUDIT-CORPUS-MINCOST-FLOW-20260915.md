# Route audit: corpus minimum-cost flow over word order

Date: 2026-09-15  
Proposal id: `corpus-mincost-flow-word-order`

## Novelty preflight

The fail-closed preflight was run before creating or running an experiment
artifact:

```text
status=novel
registered_families_checked=58
excluded_routes_checked=3
artifact=experiments/corpus_mincost_flow_word_order_20260915.py
```

The exact signature was not present in the registry. The lexical-overlap audit
also found no near-pair at the review threshold (0.40); the closest relevant
families were `variable-boundary-tape-ilp` (shared mathematical flow/equation
machinery) and `corpus-sentence-gram-fst` (shared corpus phrase-lattice
material).

## Construction review (no generator run)

The proposed state space is a directed acyclic graph whose paths are corpus
sentence/word choices. A min-cost-flow objective would select a path while
exact mirrored-character supply constraints enforce the tape. This is
mathematically distinct in name, but not in construction dimension: the flow
variables are only an alternative encoding of the same lexical arcs and
variable word boundaries already searched by `variable-boundary-tape-ilp`.
Restricting arcs to corpus phrases makes it a reimplementation of
`corpus-sentence-gram-fst`; adding POS/dependency side constraints returns to
the registered grammar/chart families. The objective changes ranking, not the
set of admissible English tapes.

Because the route has no disjoint state-space dimension, no corpus-flow
generator was run and no result is counted as evidence. This preserves the
novelty gate and avoids spending a search budget on a disguised replay.

## Concrete repair / future eligibility

To become defensible, a future route must change what is being constructed,
not just the solver: e.g. introduce a new non-lexical object with a
demonstrable invariant (a reversible discourse-level permutation or a
different tape algebra) and prove that its feasible objects cannot be
represented as paths in either existing lexical-arc family. A min-cost-flow
implementation alone is explicitly classified as a repair/ablation of those
families.


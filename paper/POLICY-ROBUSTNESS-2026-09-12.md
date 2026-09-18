# Policy robustness check for the controlled POS-pruning result

## Question

Does the controlled POS-pruning result generalize beyond the depth-first,
stable-sibling-order traversal used in the current manuscript?

## Decision

No.  The experiment supports a narrow statement about bounded depth-first
search, not a traversal-invariant efficiency claim.  Changing the two
depth-first sibling orders preserves a higher accepted-pair yield per popped
state, but changes its magnitude.  At the same 50,000-popped-state limit,
breadth-first search finds no accepted pairs in three of four arms and one in
the fourth.  The existing `5.47x` number must therefore remain explicitly tied
to its depth-first policy; it should not be the paper's central algorithmic
claim.

## Controlled design

Each condition used the same 20 opening subtrees, ordering seeds 0--19,
21,073-word frozen Brown intersection, 20--44-letter range, 18-word ceiling,
16-letter overhang limit, and 50,000-state maximum per arm.  Terminal and
incremental POS filtering were paired within every opening.  The two new
dimensions were:

- frontier: LIFO depth-first search (DFS) or FIFO breadth-first search (BFS);
- sibling order: ascending or descending stable SHA-256 word rank.

Both branches receive the same ordering whenever they reach the same state.
All four output directories pass `experiments.audit_controlled_pos_pruning`.
The traversal implementation has a small exhaustive-set unit test showing that
DFS/BFS and either sibling order return the same solution set when allowed to
exhaust a small search space.

## Results

| Policy | Terminal / incremental accepted pairs | Incremental / terminal accepted per popped state | Incremental / terminal accepted per generated state | Interpretation |
|---|---:|---:|---:|---|
| DFS, ascending sibling rank | 295 / 877 | 3.13x | 0.49x | Positive finite-DFS yield shift; worse per generated state. |
| DFS, descending sibling rank | 91 / 470 | 5.44x | 0.99x | Positive finite-DFS yield shift; magnitude depends on order. |
| BFS, ascending sibling rank | 0 / 0 | not estimable | not estimable | No accepted pair in either arm. |
| BFS, descending sibling rank | 0 / 1 | not estimable | not estimable | One incremental pair; zero terminal denominator. |

The BFS conditions are not direct evidence that the gate is harmful.  They
show why a fixed *popped-state* budget cannot be interpreted independently of
policy: both BFS variants accumulated mean frontiers of roughly 3.0--3.6
million states (about 1.1--1.3 GiB peak RSS per process) before reaching the
same 50,000 pops.  They rarely reached accepted closures at this budget.

The result does rule out a stronger reading of the original headline.  The
gate is not an expansion-efficiency improvement: the two DFS conditions give
0.49x and 0.99x accepted pairs per generated state.  It reallocates a bounded
depth-first walk toward regions with more terminally POS-admissible pairs.

## Evidence

- `runs/pos-policy-robustness-2026-09-12/dfs-ascending-v2/`
- `runs/pos-policy-robustness-2026-09-12/dfs-descending/`
- `runs/pos-policy-robustness-2026-09-12/bfs-ascending-v2/`
- `runs/pos-policy-robustness-2026-09-12/bfs-descending/`

Each directory contains frozen provenance, raw paired trials, summary,
human-readable results, and a post-run audit.  The experiment adds
`traversal` and `reverse_order` controls to the enumerator and controlled-run
driver; defaults preserve the existing DFS behavior.

## Implications for the paper

The reviewer is right about the central interpretation.  A defensible paper
could say: *A sound finite-POS viability gate changes the allocation of a
bounded depth-first palindrome search; for the recorded opening subtrees and
orders, this produces more terminally POS-admissible pairs per popped state.*
It cannot describe this as a general pruning speed-up, nor make the rate its
main evidence of NLP significance.

The next experiment worth doing is not another search-policy sweep.  To unify
the reversal and search sections, define a state-level continuation outcome on
a small exhaustively searchable vocabulary, then test whether a pre-registered
reversal-cost feature predicts exact continuation availability or search depth.
That would evaluate the paper's proposed explanatory mechanism.  A byte- or
character-level LM baseline is worthwhile only if the reversal diagnostic
remains a central contribution; it would establish robustness of that
measurement, but would not repair the missing link to search.

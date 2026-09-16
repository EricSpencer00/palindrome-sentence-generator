# Semantic relation alignment (2026-09-16)

This route pairs independently lexicalized event frames through a directed
semantic edge: preserve→recover, signal→respond, plant→grow, or
measure→adjust. The edge is construction state, not a surface word mirror.

The ledger walks whole lexical boundaries from opposite ends and emits one
matching character pair at a time. The exploration was deterministic and
exhaustive; it used no beam, MCTS, chart, CSP, ILP, static palindrome text, or
known-palindrome catalogue. The novelty preflight found no exact signature
collision against the 94-entry registry; the nearest prior was
`relation-graph-event-pair` (Jaccard 0.192308).

The run is a negative result: 36,662 states, 1,916 character-pair matches,
34,594 dead character transitions, zero exact closures, and zero mechanically
admitted candidates. The longest matched prefix was six characters. The
failure was concentrated at the first terminal lexical boundary, where the
single final place token could not satisfy a long enough reverse suffix.

The executable artifact is
[`semantic_relation_alignment_20260916.py`](../experiments/semantic_relation_alignment_20260916.py),
and the frozen run record is
[`semantic-relation-alignment-20260916.json`](../runs/semantic-relation-alignment-20260916.json).
The proposed same-family repair is to use terminal-compatible phrase spans,
precompute ≥3-character suffix compatibility, and permit a bounded odd center;
that repair was not run in this evidence package.

## Terminal-span repair

The bounded successor
[`semantic_relation_alignment_terminal_repair_20260916.py`](../experiments/semantic_relation_alignment_terminal_repair_20260916.py)
was then run as a same-family repair. It tried 12 terminal phrase spans per
frame and admitted at most one unmatched character as a centre, with the full
tape reversal recheck still mandatory. Against the current 95-entry registry,
the repair explored 646 states, tried 336 terminal spans, and found zero exact
closures or mechanically admitted candidates. Its frozen record is
[`semantic-relation-alignment-terminal-repair-20260916.json`](../runs/semantic-relation-alignment-terminal-repair-20260916.json).
The repair is attached to the base registry entry and does not add a new
construction family.

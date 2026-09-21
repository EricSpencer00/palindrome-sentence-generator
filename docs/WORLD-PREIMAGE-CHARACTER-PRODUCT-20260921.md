# Executable world preimages inside character construction

The bounded experiment recovered the existing 38-letter seed and its clause
rotation. It produced no exact palindrome above 38 letters. It is not reader
evidence and does not meet the north star.

## Actual representation change

`experiments/world_state_orbit_probe_20260920.py` checks completed plans and
then their surfaces. This experiment instead constructs a graph whose states
contain clause count, grammatical phase, number agreement, and a three-bit
resource state. Opening a gate enables delivery; closing it removes that
permission; ripping memos prevents subsequent reading of those memos.

Each graph edge emits one character. Two search frontiers move from the start
and accepting ends of this same graph, taking edges only when their characters
match. The backward frontier follows exact predecessors of forward world
transitions. It does not reverse a finished sentence or mirror a phrase bank.
The frontiers must meet at a node or one central character edge. Rendering
happens only after such a meeting. Semantic constraints are compiled before
the product traversal, rather than dynamically learned during traversal.

The frozen domain has five subject forms, seven actions, and two or three
sentences. The seed vocabulary deliberately overlaps the existing seed as a
construction calibration. This is not independent discovery of that text.
The predicate `inspire` has no modeled causal precondition, so resource
consistency must not be represented as paragraph coherence.

## Run and falsifier

Executed on `hst-bench` with Python 3.12.3, capped at 200,000 product states.
Both runs exhausted their frontier well below that limit:

| Setting | Grammar states | Character nodes | Product states | Semantic prunes |
|---|---:|---:|---:|---:|
| Executable scene | 78 | 994 | 643 | 38 |
| Preconditions disabled | 128 | 1,748 | 1,337 | 0 |

Both produced these normalized-equivalent-to-known-seed witnesses:

> an aide rips nine memos. some men inspire Diana.

> some men inspire Diana. an aide rips nine memos.

Both are 38 letters. Each passes an independent two-pointer audit and matching
forward/reverse SHA-256 hashes. Neither repeats words or sentences, mirrors
word order, contains a proper multiword self-palindromic span, or has a
self-palindromic sentence. They remain known calibration material, not novel
successes. Catalogue data was not loaded. All actual outputs and source hash
are in `runs/world-preimage-character-product-20260921.json`.

The positive control, "An aide opens the gate. The porter delivers a parcel.",
is accepted by both grammars and is 42 letters, but is not a palindrome.
"An aide rips nine memos. Some men read nine memos." is accepted only with
preconditions disabled. So is the control that closes an initially closed
gate and then delivers a parcel. These are concrete language differences
before surface completion, satisfying the operational falsifier.

Four lightweight checks passed via direct invocation of the test functions:
language ablation, independent forward acceptance and exactness of all
constructed witnesses, resource deletion, and shortcut detection. Pytest is
absent from the repository virtual environment; no pytest-run claim is made.

The product merges equivalent states and retains one witness per state. It
therefore proves bounded reachability, not exhaustive enumeration of all
surface alternatives. Absence of longer closures in this length-aware product
applies to this frozen two/three-sentence domain. Further lexical widening is
not supported by this result. The semantic ablation changes work but does not
change exact outcomes, so the experiment establishes a functioning semantic
constraint representation, not evidence of a readability breakthrough.

# Variable clause-sequence topology

`experiments/variable_clause_sequence_20260921.py` is a topology pivot from
the fixed-frame shared-character experiment. It constructs one, two, or three
complete transitive clauses, with live `and`/`while`/`as` coordination edges.
The clause count and connector are part of the NFA path; they are not appended
to a completed tape. Subject/object number and transitivity are encoded by the
authored clause alternatives.

The search applies mirrored character-domain propagation to unfinished NFA
paths and independently audits any singleton result with a second normalizer,
two-pointer comparison, and SHA-256. It never reverses a finished tape, uses
catalogue text, or repairs a near miss. The reproducible artifact is
`runs/variable-clause-sequence-20260921.json`.

The bounded run covers targets 39--100 (62 target lengths), 605 variable
sequence frames, and 250 search nodes per target. It produced zero exact
candidates and zero surviving frontiers: every target conflicted at the root
of mirrored propagation. Forward controls are retained in the artifact, e.g.
`A poet reads a map and the artist marks the letter.` (40 letters), with exact
failure and provenance recorded. Thus no readability claim is made and the
reader gate remains closed. This is a negative topology result, not a new
benchmark.

The next constructive operator is a live center nonterminal connecting two
independently licensed clause sequences, rather than another modifier or
lexical-bank expansion.

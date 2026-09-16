# Semantic insertion repair (2026-09-16)

The existing reversible-insertion family used stacked reverse lexical units as
wrappers. That is an exactness construction, but it is not a reader-worthy
sentence extension. This bounded repair required each inserted adjunct to be
independently grammatical and checked the complete sentence separately.

The preflight read all 97 retained registry families and all 6 excluded routes.
Because the state space is a repair of `reversible-grammar-insertion`, it did
not add a 98th family.

The 12 ordered pairs of distinct authored adjuncts all passed the independent
span check, but the coordinated full sentence failed the bounded frame gate.
There were no exact closures and therefore no reader-worthy candidates. All 12
near misses, including their first independent tape mismatch, remain in
`runs/semantic-insertion-repair-20260916.json`.

This result rejects the wrapper/stack shortcut rather than relaxing grammar or
catalogue controls. A future continuation needs a productive grammar with
independently licensed coordination, not more reverse lexical pairs.

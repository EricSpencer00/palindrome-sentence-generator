# Shared-character feature grammar extension

The regular shared-character propagator now includes two bounded authored
frames: a question (`can/will + singular subject + transitive verb + object`)
and a relative clause (`the/a singular subject who + singular transitive verb
+ object`). Their lexical choices are expanded through the same latent-boundary
NFA as the existing SVO frames. Agreement and valency are represented by
separate feature roles; punctuation is restored only after a lexical path is
replayed.

The run covers every target length 39 through 80 with a strict 120-node cap
per length. Differential toy checks remain enabled. Every target reaches a
root conflict before a candidate path (`nodes=1`, `conflicts=1`); shorter
targets perform 1--4 propagation rounds and remove 0--1,030 unsupported
character values. There are no exact outputs, so no prose is promoted and no
reader evidence is claimed. The existing seed calibration remains in the
grammar's prior regression family but is not used as a finished tape.

The result identifies the next operator concretely: add residual-indexed
lexical transitions to the question/relative edges, rather than increasing
the Cartesian word bank. Evidence is in
`runs/regular-shared-character-20260921.json`.

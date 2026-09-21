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

## Residual-indexed lexical branching

The follow-up adds a live residual score to each character branch: a character
is ranked by forward lexical support at the current position plus backward
support at its mirrored position. This changes only branch ordering; it never
scores a completed tape or performs repair. Targets 39--100 were rerun under
the same 120-node cap, with differential toy checks still active. All 62
targets reached a root conflict at one node and produced no exact candidate.
The next operator is an embedded object-relative edge or typed adjunct that
can add length while preserving a nonempty accepting grammar path.

The embedded edge was then added as a live ten-slot frame: an explicit object
is followed by `that`, a second singular subject, verb, and object. Targets
39--120 were rerun with residual-indexed branch ordering. All 82 targets still
reached a one-node root conflict, with no exact path or rendered candidate.
The next lexical operator is a held-out temporal adjunct on this embedded
frame, selected by residual support rather than Cartesian expansion.

The held-out temporal continuation adds `at sunset`, `before winter`, and
`during rain` as terminals inside the embedded-relative frame. They are
expanded by the NFA before mirrored support is propagated. Targets 39--140
(102 lengths) were run under the strict per-length cap; every target reached a
one-node root conflict, with no exact path or rendered candidate. The next
operator is a held-out locative adjunct or an agreement-compatible relative
pronoun variant.

The locative continuation adds fresh embedded terminals `at harbor`, `beside
quay`, and `under bridge`. Targets 39--160 (122 lengths) were run with live
residual ranking and the strict cap. Every target reached a one-node root
conflict; no exact candidate was rendered. The next operator is an
agreement-compatible relative-pronoun edge, with no change to the audit or
reader gate.

The relative-pronoun continuation expands the embedded gap with `that`,
`which`, and animate-subject `who`, while retaining the explicit transitive
valency roles. Targets 39--180 (142 lengths) were rerun under live mirrored
support. Every target reached a one-node root conflict; no exact or frontier
candidate was rendered. The next operator is a bounded complementizer/tense
variant on the embedded relative.

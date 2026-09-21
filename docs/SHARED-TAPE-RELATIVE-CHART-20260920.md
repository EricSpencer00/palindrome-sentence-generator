# Held-out object-relative shared-tape chart

This lane keeps the root-supported shared-tape solver and changes the grammar
topology rather than widening the same text-object bank. Its new production is
an object-relative noun phrase: `NPBASE RELPRON NPBASE VTRANS`, as in
“the poet that the writer reads”. The object is a bound gap, so the relative
clause is complete without copying a noun at its end. Coordination remains
recursive (`S -> C | C CONJ S`).

The known 38-letter seed is recovered only by the separate calibration grammar:
three search nodes, seven support-propagation rounds, two exact orientation
witnesses, and matching forward/reverse SHA-256 digests. It is not a result of
this held-out lane and is not promoted.

On `hst-bench`, the relative grammar was checked at 39, 40, 44, 48, 52, 60,
72, 90, and 100 letters with a 250-node cap per target. Each target reached a
root-supported contradiction after nine nodes and 29–34 propagation rounds;
there were zero exact closures. The run is a falsification of this grammar
envelope, not an English impossibility claim.

Complete prose controls remain independently parseable, including “the poet
that the writer reads studies a memo” (37 letters). No rendered row reached
the reader gate. The next constructive branch is agreement-carrying relative
subjects plus a held-out passive relative frame.

Evidence: `runs/shared-tape-relative-chart-20260920-remote.json`.

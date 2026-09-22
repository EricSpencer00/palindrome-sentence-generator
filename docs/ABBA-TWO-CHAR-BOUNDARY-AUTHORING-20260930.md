# Two-character residual boundary authoring (30 September 2026)

This probe implements a new constructive operator after the one-character
residual attempt: an intact `B2` prose clause is admitted only if its opening
two letters consume the next two characters required by the reverse of the
already-authored `A1+B1` tape. `A2` is then authored against the remaining
residual, with its role recorded. This is a live equation, not a sweep over a
finished phrase bank.

Command:

```text
python3 experiments/abba_two_char_boundary_authoring_20260930.py
```

The three fresh scene frames rendered 3 ordinary left-side controls (43, 48,
and 44 letters). Their live two-character obligations were `nr`, `ht`, and
`wo`; no ordinary authored `B2` opening satisfied any of them. Thus the run
closed 0 candidates and produced no exact palindrome. The residuals are
preserved in `runs/abba-two-char-boundary-authoring-20260930.json`, including
independent two-pointer and forward/reverse SHA audits. This is a concrete
construction failure: the next operator must select a complete ordinary word
from the residual and carry its syntactic role into `A2`, rather than merely
relaxing the two-character target or enlarging a clause bank.

All four-unit candidates are absent because no `B2` survived the hard live
boundary equation; no reader claim is made. Provenance flags exclude catalogue
reuse, finished-tape reversal, repeated self-palindromic units, and post-hoc
repair.

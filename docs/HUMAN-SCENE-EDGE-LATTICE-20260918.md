# Human-authored scene edge lattice (2026-09-18)

This lane tested a distinct construction hypothesis: choose two independently
typed miniature English scenes, and expose their normalized character equations
from the outside inward before storing a candidate. Each scene has an agent,
action, patient, and optional setting; the two lexical banks are rotated and
independently indexed. No sentence or finished tape is reversed.

## Result

The bounded run examined 2,304 independently selected scene pairs. All 2,304
were rejected by the live edge equation, with zero exact candidates and zero
mechanically admitted candidates. The longest controls are intact English-like
scene sentences, but no control is a palindrome. Human readability was not
claimed: reader eligibility remains zero.

Every row records its rendered text, independent two-pointer audit, forward and
reverse SHA-256 digests, scene provenance, and first exposed mismatch. The
independent audit uses normalized letters and compares both hash directions;
the hash equality is diagnostic and does not replace the two-pointer check.

## Concrete repair

The result shows that complete-scene pairing still exposes incompatible edges
too late. The next implementation must move the equation into a token-level
outer-inward CSP: choose agent/action/patient realizations while carrying the
residual character requirements through word boundaries, then add clause seams
and held-out role vocabulary. This preserves semantic typing while allowing
the search to construct, rather than merely test, the central tape.

Run: `runs/human-scene-edge-lattice-20260918.json`.

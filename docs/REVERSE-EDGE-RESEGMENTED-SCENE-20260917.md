# Reverse lexical edges with scene-boundary resegmentation

This lane starts with three hand-authored transitive event scenes. Each role
gets an authored, sense-compatible lexical edge (for example `pilot → keeper`
and `guides → marks`). The right realization is then generated while comparing
its consumed characters with the reverse obligation from the left scene. Word
boundaries are allowed to cross the seam; the constructor records the live
remaining debt rather than closing by reversing a completed tape.

The run rendered three complete scene witnesses and found no exact closure.
Every witness has an independent two-pointer mismatch audit, forward/reverse
SHA-256 tapes, semantic role trace, and a nonempty residual obligation. The
repair operator is a role-preserving reverse-edge substitution followed by
reopening debt at that boundary. No catalogue text, borrowed sentence,
repeated unit, mirrored word order, or self-palindromic edge is used.

Run artifact: `runs/reverse-edge-resegmented-scene-20260917.json`.

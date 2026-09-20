# Center-out lexical scene lattice (2026-09-20)

This lane grows a short semantic center (`the bell rang`, `the tide turned`, or
`the watch began`) first, then selects independently authored left and right
ordinary-English realizations. The normalized tape is audited after rendering
each growth step with a two-pointer scan and an independent forward/reverse
SHA-256 check. It uses no catalogue phrases, mirrored units, or reversed
finished sentences.

The bounded lattice contains 27 states. It is a readable scene generator and
records the best partial plus its first live obligation failure. No exact
closure above 38 letters was found in this bounded run; the next concrete
operator is a fresh boundary-indexed prepositional adjunct on the right.

Artifact: `runs/centerout-lexical-scene-lattice-20260920.json`.

# Polar-question boundary repair (2026-09-18)

This lane repairs a specific failure in the recent token decoder. The decoder
paired identical grammatical slots and stored an untyped residual, so surplus
characters could migrate to the wrong side and the right clause was assembled
in reverse slot order. The new lane uses an incoming/outgoing character pair
graph: a forward question edge is matched against an incoming answer edge, and
the recorded path is rendered in ordinary English order.

Only a small boundary-conditioned inventory was added: complete polar
questions and object-fronted declarative answers. Five additional complete
`subject met object` alternatives were then tried at the first dead frontier;
none closed. The graph also retains the 44-letter discourse line as a
diagnostic control because it exercises the useful cross-word boundary but is
not complete prose.

## Result

The graph reached one exact row:

> Was Noel an era, a gas, an item? Met in a, saga, arena, Leon saw.

It has 44 normalized letters and independent two-pointer plus forward/reverse
SHA-256 agreement. It fails exactly one strict gate,
`no_self_palindromic_proper_multiword_span`, and its answer is fragmentary.
There are no complete-answer exact rows, no mechanically admitted rows, and no
reader-eligible rows. The row is therefore retained as a geometry control, not
as a result. The failed complete-answer alternatives are preserved in the run
provenance rather than counted as a larger vocabulary sweep.

## Concrete next repair

The next construction must keep the same pair graph but expand the answer
subject/verb/object roles *before* committing the whole phrase, carrying the
dead frontier's character debt across those token boundaries. Preserve the
lexical history and reject proper palindromic subspans before any reader
package is created. This is a boundary-conditioned grammar repair, not a
larger vocabulary sweep.

Human readability remains unmeasured; programmatic exactness and gate checks do
not certify English prose.

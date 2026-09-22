# Open-residual ABBA paragraph cycles

## Decision

Use the deployed v3 API's compositional idea, but move the reusable boundary
one level down. The API returns to an **empty** character obligation after each
mirror-pair. That makes length cheap, but it also makes the output a nesting of
already closed units. The paragraph generator should instead return to the
same **nonempty** character obligation after one grammatical discourse step.

The target state is a product of:

- the left paragraph grammar (`A -> B`);
- the right paragraph grammar traversed outside-in (`A' -> B'` during search,
  rendered normally as `B' -> A'`);
- agreement, valency, discourse-referent, and paragraph-phase state; and
- one owned, nonempty character residual.

A reachable and coaccessible strongly connected component at the same full
state is a scalable construction. Replaying the cycle adds prose on both sides
while preserving the open seam. Exact closure is permitted only on the exit
path after both grammars reach their terminal states.

## Why this differs from the live API

On 2026-09-22, `/api/v3/health` reported 540 bank entries, 499 generated
entries, 41 catalogue entries, and 14,500 letters of novel compositional
capacity. A deterministic 200-letter request returned 174 exact letters in 11
distinct chunks, but the rendered text began “Pro, cat estimates am. Meet at
seat...” and remained visibly fragmentary. Independent normalization and a
two-pointer/hash replay confirmed exactness. This is capacity evidence, not
readability evidence.

The reusable idea is the compositional state and provenance trail. The parts
we do not promote are the preclosed mirror-pair bank, catalogue fallback, and
length claims detached from human readability.

## Implemented certificate

`llm_palindrome/recursive_product.py` now:

1. consumes left words in reading order and right words outside-in;
2. carries unmatched characters across different word boundaries;
3. rejects an empty residual before both grammars terminate;
4. computes reachability and coaccessibility in the constrained product;
5. finds strongly connected components containing only nonempty-residual
   states; and
6. reconstructs `prefix + cycle^k + suffix` witnesses deterministically.

The synthetic certificate has the recurrent residual trace
`left:b -> right:c -> left:b`. Pump counts 0 through 4 produce independently
verified exact tapes of 10, 14, 18, 22, and 26 letters. The left and reflected
right word-boundary signatures differ. Four focused tests cover the pump,
coaccessibility, shifted boundaries, and intermediate-closure rejection.

This is deliberately **not** an English result. Its reader packet is empty and
the 38-letter sentence remains the only reader-worthy anchor. The certificate
establishes the missing scalable algorithmic primitive so the next experiment
can ask a clean linguistic question: does an authored ABBA discourse grammar
contain a natural coaccessible open-debt cycle?

## Staggered paragraph seams

The paragraph version is not four closed palindromic units.  It is a dual parse
of one half-tape: the forward parse contains complete sentences `A` then `B`,
while the reverse-facing parse contains independently generated sentences
`B'` then `A'`.  Their sentence boundaries are variables.  A closure is
cross-sentence only when the forward boundary and the reflected reverse
boundary occur at different letter offsets, no complete sentence mirrors
another complete sentence, and no proper contiguous sentence block is itself
a palindrome.

`llm_palindrome/paragraph_product.py` implements that audit and the associated
online word-residual product.  A 46-letter inherited formulaic tape can be
repunctuated to exercise the staggered geometry, but remains a rejected
topology control: punctuation did not create new letters, and the central
admission gate still detects its formulaic symmetry.  The first fresh typed
four-sentence grammar explored 49 states and 69 transitions, reached six
matched outer letters, and produced no exact closure.  That run is a bounded
endpoint test, not evidence against the paragraph method.  Its concrete next
change is structural: compile alternative complete-clause paths and movable
sentence boundaries into a packed automaton instead of widening the same SVO
word lists.  Artifact:
`runs/staggered-abba-paragraph-product-20260922.json`.

That packed follow-up is now implemented in
`experiments/packed_staggered_paragraph_automaton_20260922.py`.  Four complete
clause shapes per discourse phase increased the reachable product from 49 to
227 states and the maximum live residual from six to nine characters, but no
path reached either second-sentence state.  The failure is therefore an entry
path problem, not a reason to enlarge the nouns and verbs again.  The next
operator must induce grammatical outer paths that arrive at `A -> B` with a
live residual, then hold those paths fixed while testing the inner sentences.
Artifact: `runs/packed-staggered-paragraph-automaton-20260922.json`.

## Natural cycle diagnostics

A streamed Brown-corpus shape mine found two ordinary open-residual equations:
`no name` / `one man` preserves residual `name`, and `no race` / `one car`
preserves residual `race`.  These phrases are diagnostic grammar shapes, not
generated candidates.  Replaying either literal phrase would repeat content
and is forbidden; a scalable prose grammar must realize the same state cycle
with fresh lexical choices.  A targeted search found no compatible Brown
cycle at the live residuals of the 38-letter anchor, and a first fixed-position
clause grammar failed before reaching either NP anchor because its outer
adjuncts were incompatible.  The paragraph product therefore moves the
sentence seam itself rather than enlarging those same lexical banks.

## Promotion gate

The next grammar run is promoted only if it yields all of:

- a coaccessible cycle whose residual is nonempty at every internal state;
- independently parsed `A/B` and `B'/A'` prose with shifted word or sentence
  boundaries;
- no repeated lexical unit, boundary-aligned word-order mirror, catalogue
  text, post-hoc repair, or proper intermediate palindromic span;
- at least one exact rendered candidate longer than 38 letters; and
- a blinded intact-versus-shuffled reader packet before any readability claim.

If the grammar has no such SCC, the first unsupported product frontier—not a
larger duplicate sweep—determines the next grammar production to add.

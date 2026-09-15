# Parallel constructive search queue

Three independent Luna branches were run against the same non-negotiable
target: a novel, intact, letter-level English palindrome longer than the
38-letter control.  Every branch preserved its rendered outputs and exact
audit; none was treated as readability evidence without human readers.

## Typed lattice / lexical-family diversity

The live syntax/residual search used 131 typed plans and rotated exclusions
over the control's verb, object/number, person/name, and all-content families.
The 1,000-seed run (beam 1,200; 220 steps) emitted 767 exact closures, but only
one unique closure: the known control.  Novel closures at or above 39 letters:
zero.

## Syntax-first clause pairing

The clause enumerator generated 1,383,480 complete typed clauses before
matching reversed character tapes; the boundary-residual trie contained
5,107,224 nodes.  It found zero exact clause pairs.  A follow-up role-
preserving subject/verb/object/adjective/place substitution pass generated
52,700 repaired clauses and still found zero exact closures; its largest
residual match was one character.

## Typed semordnilap inventory

The reverse-pair branch searched 291 frequent-word pairs in typed POS/valency
frames and found 14 exact closures above 38 letters.  The longest rendered
closure was:

> Lager parts diaper smart; trams repaid strap regal.

Its independent tape is 42 letters and exactly palindromic, but the admission
gate rejects it (lexicon failure, word-order symmetry, and a proper
self-palindromic span), and blinded readers have not seen it.  It is retained
only as a failed construction trace.

## CFG/feature lattice and bidirectional transducer

The CFG branch produced 52 exact lexical closures, but the only two that
passed its mechanical floor were 32 letters long and still fragmentary:
`dessert set one man name note stressed` and its reverse ordering.  It found
no novel ≥39-letter readable output.

The independent bidirectional typed transducer indexed complete adjective /
subject / verb / object clauses in a character trie.  Its bounded run found
zero exact ≥39-letter closures; the strongest retained near miss was
`Teacher sees. Calm pilot sees a friend.` (31 letters, 20 mirrored-character
mismatches).  The transducer's next operator is reverse-trie-compatible slot
expansion with the same joint syntax checks.

## Next constructive move

The queue has therefore exhausted lexical rotation and one-at-a-time repair.
The next operator is coordinated role-preserving substitution at the same
boundary, with number agreement, verb valency, and adjective attachment checked
after each complete clause before the residual trie is resumed.  The paper and
API remain gated until a novel closure passes the mechanical gate and a blinded
intact-prose versus shuffled-control study.

Commits: `2fdb450` (lexical-family diversity), `626bdfd`/`a7c231e`/`e5786c1`
(syntax-first, boundary trie, and residual repair).

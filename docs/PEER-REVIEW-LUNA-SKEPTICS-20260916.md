# Three skeptical Luna reviews (2026-09-16)

Three independent Luna passes reviewed the ten requested construction lanes,
the subsequent continuation lanes, the aggregate audit, and the reader
gate. They were asked to find shortcuts or unsupported claims, not to defend
the current result.

## Orthogonality and novelty

The reviewer found no duplicate IDs, signatures, or run-artifact references
among retained families and agreed that the explicit catalogue/repetition
controls are correctly quarantined. It did identify that the aggregate report
was not itself represented in the novelty registry. The registry now has an
`audit_reports` section pointing to
`runs/parallel-luna-readability-diagnostics-20260916.json`, explicitly marking
it as a report rather than a construction lane. Historical checkpoint counts
in the evidence ledger are labeled as checkpoints; the closing snapshot is
the current 287-artifact / 31-exclusion registry.

## Exactness and provenance

The reviewer independently replayed the selected tapes and confirmed the
recorded lengths and non-exact pointer/hash results for the ten lanes and
continuations. It found one audit defect: the agreement-morphology run had
hashed raw rendered text and supplied no reverse digest. The transducer now
records normalized forward and reverse SHA-256 digests plus their equality, and
the contract test recomputes both from the rendered tape.

## Reader study and paper claims

The reviewer found the paper and API correctly fail-closed: no automatic score
is presented as readability evidence and no candidate reaches a reader packet.
It found three builder gaps, now covered by tests and implementation:

1. Candidate provenance must be a structured immutable record with generator
   and source hashes plus explicit false flags for copied/borrowed text and
   shortcut symmetries.
2. Intact controls must match the candidate within five normalized letters.
3. Shuffled controls are regenerated until their text and normalized tape
   change and their normalized tape is not itself a palindrome.

The reader package remains `ready_for_human_collection` only after an exact,
mechanically admitted candidate is supplied; no current near miss is promoted.

## Current disposition

The reviews did not produce a reader-worthy palindrome. Their concrete repairs
are committed, and the constructive goal remains active: every next failed
generator must introduce a new construction state or a targeted repair rather
than a larger duplicate sweep.

The closing queue pass added three non-overlapping Luna states: constrained
meaning-preserving edits, append-preserving clause algebra, and discourse
relation involution. Their 20 rendered prose rows are included in the latest
aggregate (4,770 rows / 210 route phases), with independent audits and named
repairs; none is exact or reader-eligible.

A second repair pass added typed semantic center-out construction, lexical
word-equation intersection, and joint slot/boundary resegmentation. The
reviewers required that malformed outputs be removed from the prose claim; the
slot lane was repaired from determiner duplication to nine complete 45--62-
letter scenes. The three lanes remain non-exact, independently audited, and
reader-ineligible.

The newest continuation queue adds endpoint-aware bilateral seam decoding
(six 100--105-letter scenes), mutable fresh-scene tape/CFG resegmentation
(123- and 126-letter complete scenes), and typed semantic-slot repair (six
109--160-letter scenes). Each was novelty-preflighted before execution and
keeps independent pointer/SHA validation, provenance, and a named next repair;
none closes exactly.

A subsequent constructive pass added a paired semantic CFG, a joint
semantic/inflection boundary DP, and an online clause-order scene lattice.
They contribute 108-, 109-, and 151-letter intact prose states with
independent audits, provenance, novelty checks, and concrete repairs; none is
exact or reader-eligible.

The follow-up queue added two more non-overlapping Luna states: a typed
character-semantic beam (12 complete 109--115-letter scenes) and a fresh
CFG/Earley character intersection (two complete 85--137-letter scenes). Their
rendered prose, independent pointer/SHA audits, provenance, novelty checks,
and typed next repairs are preserved; both remain non-exact and
reader-ineligible.

The current continuation adds five orthogonal Luna states: nested-free
clause-boundary DP, semantic phrase-edge joining, coupled object/attachment
repair, role-typed semordnilap clause products, and a typed reversible-clause
composer with appendable frame growth. They add 19 rendered 92--124-letter
scenes to the aggregate, each with independent pointer/SHA validation,
provenance, novelty preflight, and a concrete next repair. None is exact or
reader-eligible; the constructive search therefore continues.

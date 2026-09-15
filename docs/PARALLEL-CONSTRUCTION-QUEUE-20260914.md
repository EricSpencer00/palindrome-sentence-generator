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

## Whole-discourse order and boundary witnesses

The whole-discourse branch removed the earlier pairwise-clause assumption.  It
searched ordered subsets of 60 newly authored, intact same-scene sentences and
allowed a character residual to cross any number of sentence boundaries.  The
frozen run explored 63 states and all 1,140 ordered outer-sentence pairings,
but no pairing reached a live second-sentence transition.  It produced zero
exact closures.  Fifteen 100--160-letter intact controls are preserved with
independent audits; the strongest garden control was:

> Flowers attract bees. New leaves cover the branches. An observer waits. Some gardeners dig. Clean boots rest beside the door.

That control has 101 letters and 39 mismatched mirrored pairs.  It is ordinary
rendered prose, not a palindrome candidate and not reader evidence.

A follow-up constituent bridge jointly varied the opening and final complete
clauses before freezing them.  It found 24 genuine nonempty residual witnesses
and tested 984 orders containing up to three additional complete same-scene
clauses.  None closed exactly.  This rules out adding more permutations of the
same frozen bank; the next branch must expand grammatical phrase realizations
while a residual is live.

## Dual typed chunk residuals

The dual-parse branch generated 50,000 complete NP/VP/PP sentence parses and
intersected their tapes incrementally through a reverse character trie.  The
replay traversed 2,381,329 residual characters (maximum sentence length 59)
and found zero exact closures above 38 letters.  A synthetic test confirms
that the kernel can match staggered chunk boundaries.  The remaining missing
operation is a boundary-state grammar automaton that can emit a newly licensed
NP, VP, or PP attachment before either side has completed its sentence.

A separate exact residual solver now performs that emission over 12 complete
typed plans, including numbered, adjectival, adverbial, imperative, relative,
and coordinated clauses.  It consumes words from opposite sentence edges,
cancels their letters immediately, and carries a residual across staggered
word boundaries.  All 144 plan pairs completed without exhausting their state
budgets (1,706 states total).  The only exact closures were the 38-letter
control and its clause-reversed rendering; there were zero mechanically
eligible closures at 39 letters or longer.  Its next expansion is therefore
residual-indexed lexical mining: add only agreement- and valency-safe words or
multiword constituents whose letters extend an observed live residual, rather
than enlarging every slot indiscriminately.

That residual-indexed replay is now implemented.  It recorded 1,409 reachable
nonempty residual states and found 27 compatible additions across seven typed
roles (names, plural verbs, singular and plural arguments, prepositions, and a
multiword person constituent).  Replaying all 12 plans with only those mined
additions explored 2,359 states; no pair exhausted its budget.  The result
again contained only the 38-letter control and its reversed clause order, with
zero mechanically eligible closures.  The strongest unresolved seams are
`ne` after `an aide rips nine` / `inspire Diana`, `ni` after `an aide rips` /
`inspire Diana`, and `rimda` after `an aide` / `admire Diana`.  The next
operator must therefore expose optional typed attachment slots at those live
seams; another flat word-list expansion cannot consume them.

The strongest seam was then isolated as a direct grammatical phrase equation.
Holding only the productive `Diana` / `an aide` endpoint factor fixed, a hash-
indexed search joined 174,002 Brown-derived object NPs and 13,516 human-subject
NPs across 181 singular and 181 plural verb forms.  The solver visited 791
live residual states, exhausted no budget, and recovered exactly one surface:
the 38-letter control.  There was no novel closure.  This eliminates a large
predicate/subject phrase space around that single endpoint; the next search
must jointly vary the terminal name and grammatical opening NP rather than
continue enlarging the already-exhausted middle bridge.

## Explicit shortcut rejection

The superficially attractive extension

> Now an aide rips nine memos. Some men inspire. Diana won.

is an exact 44-letter palindrome, but it is rejected: the original 38-letter
palindrome remains an intact proper multiword span inside it.  This is precisely
the forbidden self-palindromic-wrapper construction, so neither its extra six
letters nor its fluent clauses count as progress.

## Next constructive move

The queue has therefore exhausted lexical rotation, complete-sentence tape
pairing, frozen sentence permutations, and one-at-a-time repair.  The next
operator is a grammar-state residual automaton: jointly emit typed multiword
constituents on both sides, allow their word and sentence boundaries to stagger,
and check number agreement, verb valency, and attachment while the character
residual is still live.  The paper and API remain gated until a novel closure
passes the mechanical gate and a blinded intact-prose versus shuffled-control
study.

Commits: `2fdb450` (lexical-family diversity), `626bdfd`/`a7c231e`/`e5786c1`
(syntax-first, boundary trie, and residual repair), `69f5a4d` (whole-discourse
ordered subsets), `a424f02`/`5228430`/`196afb8` (dual typed chunk searches), and
`c0dd732` (witnessed boundary bridge), and `810a97c` (exact dual-plan residual
search with relative and coordinated expansions), and `bc0f1eb`
(residual-indexed role-safe lexical expansion), and `f195f8d` (large indexed
predicate/subject phrase bridge).

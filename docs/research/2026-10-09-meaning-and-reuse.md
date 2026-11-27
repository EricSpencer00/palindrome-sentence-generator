# Longer palindromes without padding

The goal is longer exact text with clear, connected meaning. Repeated words
are allowed when they keep track of people, objects, actions or consequences.
Repeating a sentence, renaming its actor, or paraphrasing the same assertion
does not establish new meaning. Extra length cannot compensate for worse
whole-text grammar or meaning.

## What can repeat

Names, pronouns, articles and referring noun phrases can recur. A later event
may need the same person or object as an earlier event. Predicate phrases may
recur when their realized role, referents, speech act or useful consequence
changes. A question followed by an answer can make a different contribution;
asking the same question repeatedly cannot.

The experiments use explicit bindings for these prechecks. Connected-event
search tracks who holds, reads and signs each document, including the actual
before/after state. Typed-clause search records canonical agent/action/object
facts. The free-start trial distinguishes questions from statements. They
reject repeated identical uses and retain reference words. These finite
symbolic checks do not prove that a paragraph is interesting or coherent.
Changing a name alone can pass a binding check while still failing whole-text
meaning review.

## Real hurdles and imposed limits

Mirrored letters are a real constraint. A word's spelling can force several
words elsewhere, with different boundaries. The relevant question is whether
those words work in context, rather than whether the first word is long.
Measuring meaning is another real hurdle: local word likelihood and entity
counts can improve without producing a coherent passage.

Vocabulary, sentence templates, fixed centers, word reuse bans, beam width,
overhang caps and early stopping are choices made by the implementation. A
failure within one such language is not a failure of English generally.
More compute helps only when supported alternatives were pruned or unfinished;
it cannot create an alternative absent from the language.

The first October 9 batch illustrates both problems. All 63 event scenes
exhausted their available paths before a second event. The contextual search
produced exact text up to 252 letters by recycling the same two assertions.
The model's 24 proposals and feedback repairs produced no exact palindrome.
These are finite-domain results, not impossibility proofs.

## Avoid local minima

Compare different constructions, not just nearby score weights. Preserve the
old method as a control; allow fresh starts and resynthesis; retain distinct
grammar and meaning hypotheses; record why branches fail. Keep the best
meaningful text separately from the longest mechanical palindrome.

An invariant copied center and a locally attractive stock phrase can trap the
search. The new free-start trial removes the fixed center, admits independent
clause boundaries, allows legitimate entity reuse, and checks complete clauses
and a connected fact graph. A larger shared-word beam alone is a separate
control, rather than evidence of a new search method.

`beam_search` now exposes its historical 24-letter overhang cap and stopping
after an incumbent when all remaining states exceed three times the minimum
length. Defaults remain compatible. The free-start trial uses a 96-letter
overhang, disables that incumbent stop, raises the beam from 96 to 384 and
allows up to 2,500 letters and 1,500 steps. Deadlines and explicit reuse checks
remain. These raised ceilings do not certify coverage of all English.

## Measurement

Validate the whole normalized letter tape with two independent implementations.
Keep invalid and interrupted proposals, actual inputs, hashes, reference
matches and source credits. Separate complete extension grammar from complete
passage grammar. A model's description of its own plot is not a meaning label.

Use blinded whole-passage reading after a frozen intact-versus-shuffled
calibration. The current machine screen passed its narrow calibration
(8/9 intact and 0/9 shuffled at least 2/3 on both dimensions). None of the
initial 10 exact texts scored 3/3 on both grammar and meaning. This is a machine
screen, not a human study. The target remains fresh readers accepting the full
passage, with meaning retained as it grows.

The [Argonne run record](../../runs/argonne-length-meaning-20261009/README.md)
contains the completed measurements, raw failures and reproducible sources.

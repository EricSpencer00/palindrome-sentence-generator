# A search designed for readable, original palindromes

The target is **the longest original palindrome that still says something
clearly**. Exactness and provenance are hard gates. Grammar and meaning have
minimum acceptable levels; extra letters cannot buy their way past those
levels. Keep the tradeoffs visible rather than add readability, length and
novelty into one arbitrary score.

This is a proposed generator, not an implemented replacement or a validated
optimum. A bounded solver can establish exactness and search completeness for
its specified language. It cannot establish that its language covers every
good English palindrome, or that a learned readability score matches readers.
There is no demonstrated algorithm that maximizes all three goals universally.

## Choose words by what they force the other side to say

A long word is not automatically bad. Its reversed letters may allow a natural
phrase, several awkward fragments, or no grammatical continuation at all.
Evaluate those actual alternatives in their sentence context before choosing
the word. Do not reward it merely for adding letters, or reject it merely for
being long.

For example, [“Put rats at a startup”](https://www.reddit.com/r/palindromes/comments/1wpt1zb/put_rats_at_a_startup/),
posted by [u/whenindrime](https://www.reddit.com/user/whenindrime/), is exact:
`startup` mirrors `Put rats`. Seven letters have a compact two-word reading.
The poster is credited; original authorship is not independently established.
This reference is not a newly generated result.

The [corpus auditor](../../tools/audit_credited_palindromes.py) measures which
words overlap each reflected letter span. That describes the constraint; the
overlap count is not a readability score. A six-word natural phrase can be
better than two nonsensical words. Boundaries may also cross the edge of the
word under consideration, so checking each reversed word in isolation is
insufficient.

## Representation and exact search

For a target of `n` normalized ASCII letters, keep one tape `x` with the hard
equations `x[i] = x[n - 1 - i]`. Reject unsupported non-ASCII alphabetic input
for separate review. Spaces, case and punctuation contribute no letters.

Compile a bounded family of ordinary-English realizations into a shared
character graph or packed grammar chart. Search state carries:

- Supported forward and backward parse frontiers, including partial words.
- Agreement, verb arguments, clause boundaries and unresolved dependencies.
- Entities, their references, and the propositions already expressed.
- Remaining length, supported letter domains and the selected intent family.
- Enough history for the stated novelty and repetition checks.

Every retained arc must have support in a complete start-to-finish derivation
under the mirrored equations. Propagate grammar and mirror constraints together
before branching. In a graph formulation, the two frontiers traverse matching
character edges and must join on one shared path, at a node or central edge.
In a chart formulation, require complete root support, not just locally
compatible edges. Either formulation permits different word boundaries on the
two sides and a center inside a word.

Punctuation is a grammatical choice: a question, address or ordinary sentence
can license a different reading of the same letters. Explore these choices
with the parse, then render the selected interpretation and independently
verify that rendering changed no normalized letter. Punctuation cannot supply
a missing verb or an unexpressed relationship.

For each proposed word or span, obtain the supported opposite continuations
in their actual grammar and discourse state. Estimate their naturalness and
the argument obligations they leave. Rank the joint continuation, not the
visible word alone. Zero *complete structural support* is a sound rejection
within the bounded grammar; a low learned quality estimate is only a heuristic.
Keep several supported readings rather than freeze the first segmentation.

## Meaning and language coverage

Start with several flexible, concrete intents. Each contains a small set of
entities and propositions, such as an agent performing an action on an object.
Allow alternative phrasing, word order, questions and discourse structure.
Neither one fixed sentence nor one tiny hand-written frame bank should define
the whole search space.

Proposition coverage requires an actual role-bearing interpretation: the text
must assert who did what to whom. Seeing the three words somewhere is not
coverage. Subject and object reversal, unsupported pronouns, missing arguments
and disconnected events should remain visible failures. Parse constraints can
enforce selected obligations; automated entailment and coherence judgments
remain estimates requiring reader validation.

Use credited short successes and ordinary prose to propose reusable lexical
domains, paraphrases and grammatical patterns. A model proposes expansions to
the language; the exact solver owns the letters. When support vanishes, record
the first conflicting grammar/letter obligation and expand the relevant domain.
Require deeper supported continuations on held-out intents before scaling that
expansion. Merely moving a mismatch by one letter is not a readability gain.

## Grow without trapping the search inside a bad seed

Maintain diverse short exact candidates that pass the current quality screen.
Preserve separate families of syntax, meaning and boundary patterns; a thousand
near-identical variants are not a thousand independent possibilities.

Attempt bounded replacements of mirrored spans with both sides solved jointly.
The edit may move word and sentence boundaries. Reparse and rescore the complete
result, including the seams, every time. Accept additional length only while
the quality floor and novelty conditions remain satisfied. Keep the shorter
parent when an extension fails.

Also retain full reconstruction from a different intent, grammar family or
target length. Always extending one frozen tape can make its inherited flaws
inescapable. Appending palindromic blocks or wrapping a seed is not evidence of
connected prose.

```text
freeze corpus split, quality thresholds, baselines and search budget
propose several intent families and broad realizations
for target_length in scheduled_length_bands:
    chart = compile(realizations, target_length, grammar_and_role_obligations)
    propagate(full_parse_support, mirrored_letter_equalities)
    while budget remains and supported alternatives exist:
        branch on a supported unresolved character or grammar choice
        rank choices by both contextual readings; preserve family diversity
        if a complete derivation closes:
            render punctuation; independently verify letters and exactness
            check attribution, copied passages, grammar and expressed meaning
            retain eligible candidates on the length/quality/novelty frontier
    expand a diagnosed domain or try a new family when support is exhausted
try bounded paired-span growth; rebuild when existing tapes constrain progress
freeze selected outputs; evaluate with fresh blinded readers
```

Beam limits or learned pruning make this an approximate search. Report that
explicitly. Only exhaustive traversal of the stated finite domain licenses a
bounded no-solution claim; a timeout or empty beam does not.

## Use the feedback and corpus honestly

Keep exactness separate from reader ratings. Exclude non-palindromes from
positive construction examples while retaining their original ratings and
control status. Scalar ratings are useful supervised feedback; they do not
automatically become pairwise preferences or a completed RLHF training run.

Keep Reddit post links, poster names and the original rendering with every
example. Mark copied or adapted text as such. Corpus retrieval must never be
reported as independent discovery. Deduplicate normalized tapes and group
related constructions before splitting data. Keep an author's posts, source
threads, paraphrases and shared construction families together across the
development/validation boundary, resolving intersecting groups jointly.

Novelty means independently produced content relative to the declared corpus
and provenance checks; it cannot certify absence from all prior writing.
Ordinary repetition of an entity can help coherence. The repository's stricter
construction gates remain in force unless explicitly revised and versioned;
they are not definitions of human readability.

## Build on the existing failures

- [Overhang lookahead](../../llm_palindrome/overhang.py) already counts possible
  repayments. The missing distinction is the quality and complete grammatical
  support of the repayment in context.
- [Semantic-plan coverage](../../llm_palindrome/semantic_plan.py) currently
  checks term membership. It cannot establish an asserted relation.
- [The shared-tape chart](../SHARED-TAPE-SUPPORT-CHART-20260920.md) already has
  free word boundaries and full-root support. Its small grammars yielded no
  new longer candidate. Renaming that method does not broaden its language.
- [The world-state product](../WORLD-PREIMAGE-CHARACTER-PRODUCT-20260921.md)
  already combines exact character search with grammatical and resource
  obligations; it recovered the inherited 38-letter seed and rotation only.
- [The endpoint audit](../ENDPOINT-FEASIBILITY-CERTIFICATE-20260921.md) shows why
  changing interior word choices cannot fix incompatible exterior domains.
- [The resource study](../../experiments/DIRECTIONAL-SEARCH-RESOURCE-SENSITIVITY-20260925.md)
  found “one poem association” in 36 of 66 selected outputs. Exactness and
  lexical admission did not make those outputs coherent.
- [The readability audit](2026-10-08-readability-validity.md) explains why
  Brown/GPT scores and longer exact constructions are not human-quality gains.

The new hypothesis is that broader learned language support, contextual
reverse-completion ranking and relation-based feedback can improve actual
readability together. It remains untested.

## One bounded test before scaling

Freeze one baseline and compare ablations of contextual reverse completion,
relation-based feedback and flexible family expansion. Use matched target
lengths, corpus partitions, seeds, search budgets and output-selection rules.
Record actual expansions, model calls, time and every failure; compare both
coverage and quality among successes. Nominal beam width alone does not equalize
search opportunity.

Calibrate grammar and meaning thresholds on development controls, then freeze
them. Fresh blinded readers rate grammar, clear meaning and naturalness, and
paraphrase the main event. Include intact prose, malformed text, shuffles and
credited existing palindromes; hide origin during rating. Report uncertainty
by construction family and keep the thresholds unchanged after unblinding.

The primary result is the longest tested band that clears both reader-quality
floors while retaining original exact outputs at a reported success rate.
Also report shorter high-quality outputs and the complete tradeoff frontier.
Do not stop after selecting one lucky long survivor.

Before the run, fix its resource cap, minimum useful quality improvement and
acceptable coverage loss. Stop scaling a mechanism if its bounded test shows
no meaningful reader gain, its uncertainty remains unresolved, or its coverage
falls past that limit. Gains in support depth, model scores, length or phrase
diversity alone do not pass this gate. An inconclusive result calls for a new
explicit experiment, not a claim of perfection.

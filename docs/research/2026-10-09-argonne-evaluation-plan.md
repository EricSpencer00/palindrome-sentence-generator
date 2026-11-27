# Quick Argonne evaluation: length versus readable meaning

The goal is to generate original, exact letter palindromes that read as clear
sentences and eventually connected paragraphs, extending beyond 1,000 letters
without losing meaning. Phrases, uneven word boundaries, partial-word overhangs
and punctuation are allowed. Success means longer understandable writing, not
merely more exact letters or higher model scores.

The quick construction screen was submitted as Polaris job **7731461** under
`EVITA` on October 9. It requests one debug node and ten minutes. The coordinator
verified the MacBook-origin SSH route, live account membership, scheduler and
Python 3.11.15. Home-directory staging hit quota before submission; the job
uses the existing Eagle project storage, with `home:eagle` requested. The
failed task-created home workspace was removed. The job finished with
`Exit_status=0`, using 99 seconds on one node after 9 minutes 21 seconds in the
queue. All 72 outcomes, all returned letter tapes and all input/source hashes
were retrieved and independently checked. Task-created remote scratch was
removed after verification; the unrelated held job was preserved.

The [completed results and interpretation](../../runs/argonne-length-readability-20261009/README.md)
show 43 selected exact outputs, 33 distinct selected tapes and a longest
observed closure of 447 letters. No output reached the two highest bands in
this budget. A separate blinded model-agent screen scored all 67 distinct
selected/longest texts at meaning 1, with grammar 0 or 1. This is development
feedback, not human readability evidence or a universal length limit.

The first long target is **1,000+ normalized letters**, with clear grammar,
meaning and connected discourse. The full paragraph's letter stream must be a
palindrome; its sentences need not be individually palindromic. Phrase, word,
sentence and paragraph boundaries can differ between mirrored positions.
Overhangs can cross those boundaries. The intended reader target is 3/3 for
grammar and whole-text meaning; a diagnostic score of 2 is an intermediate
screen, not completion of that target.

## What exists and what remains

| Capability | Implemented evidence | Remaining requirement |
|---|---|---|
| Exact letter symmetry | `search.py`, `centerout.py`, independent verifier | Keep it a hard gate on every full output. |
| Unequal word boundaries and overhangs | `WordTries`, `consume`, two-sided state | Evaluate the grammatical cost of repaying the debt in context. |
| Multiword phrase units | `phrases.py`, `unit_letters`, `CoherentScorer` | Check whole-text grammar and joins; ordinary phrases may be reused, copied complete examples are credited. |
| Long exact output | Historical 568/666-letter lineages; October 8 probe reached 804 letters | Neither length nor exactness establishes readability. |
| Short external examples | 189 exact Reddit titles, up to 137 letters | They have attribution but no collected readability labels. |
| Reader feedback | 183 scalar labels preserved | Their item IDs still lack a verified passage join; do not train or score passages against guessed matches. |
| Readable length growth | Short reference constructions and many exact extensions | No established reader-validated long construction frontier. |
| Meaning-aware construction | Bounded grammar/world-state products | Broader supported realizations and actual relation coverage remain proposed work. |
| Beyond 1,000 letters | Exact-construction machinery and legacy long assembly diagnostics | Show fresh complete texts, intact meaning and originality; never promote assembly length alone. |

The proposed [readable-search design](2026-10-09-readable-search-design.md) is
not the algorithm this quick run will validate. The run establishes an honest
baseline for the existing components and exposes where they fail.

## Smallest useful construction screen

Use one bounded CPU allocation and **72 jobs**: two existing search directions,
two inventories, six length bands and three fresh seeds. Freeze the full
manifest before search. The inventories are words only and words plus the
2,000 most frequent eligible attested pairs, with a phrase-aware scorer.
Neither arm may retrieve complete
Reddit palindromes as generated results. This compares the two implementations
descriptively; their different closure semantics prevent an isolated causal
claim about direction.

| Target letters | Main question | Available Reddit references |
|---|---|---:|
| 30–79 | Can the system produce a clear short thought? | 53 |
| 80–159 | Can it sustain a complete sentence? | 1 |
| 160–319 | Do additional clauses preserve meaning? | 0 |
| 320–639 | Are the sentences connected? | 0 |
| 640–999 | Does coherence survive paragraph length? | 0 |
| 1,000–1,499 | Can it cross 1,000 without padding or collapse? | 0 |

There are another 135 exact Reddit references below 30 letters. These counts
describe the [collected corpus](../../data/reddit-palindromes/2026-10-09/README.md),
not quality judgments or an exhaustive sample. Long prose controls must come
from a separate source; repeating short palindromes does not make a valid
long-readability control.

The prepared runner fixes 20 seconds of cooperative search per job,
25-second hard process deadline, beam 48, candidate menu 200, parent quota 8,
overhang cap 24. Allow 1,024 flattened words and 1,500 expansion steps so the
previous 256-word/400-step caps do not silently censor the longest band.
The letter cap still bounds each result. These are exploratory settings, not
previously established optima. Record actual work, setup time and cap hits.

Use the same seeds and resources in each paired cell; randomize execution
order. Reuse initialized resources within workers. Request only one short
allocation and stop at its fixed wall limit. Queue delay, dependency setup and
runtime must be reported separately. Three seeds locate obvious failure modes;
they cannot establish a general method ranking.

The two inventories form an explicit phrase ablation. Hold the scorer's
constituent-word treatment and all other settings fixed. The runner corrects
the existing rank scorer's unit-based repetition count so that a phrase cannot
hide repeated constituent words. This is a documented baseline adaptation,
not an evaluation of the unmodified historical scorer.

The [prepared runner](../../tools/polaris/length_readability_quick.py) builds
resources once, then supervises at most 12 forked processes. It saves each
selected result and the longest observed exact closure separately, preserving
partial output on a hard deadline. Full strict admission is attempted only up
to 128 flattened words; larger texts explicitly record it as unrun. Exactness
and the target band are always checked. The tiny offline tests do not run the
full grid or consume a cluster allocation.

Implementation cautions from the existing code:

- Reuse the search APIs; the legacy `llm_palindrome.generate` CLI is retired.
- Flatten phrase units for word/repetition counts; count normalized letters
  rather than `len(unit)`, which includes spaces.
- The October 8 runner counts units as words and raw unit lengths as letters;
  it needs adaptation before phrases enter its inventory.
- Outside-in search currently hardcodes debt length 24; center-out exposes it.
  Keep those aligned for this screen and record the limit.
- Keep all closure, timeout, no-output and execution-error records. One selected
  text per successful job is not the same as the complete search population.
- Preserve individual mechanical checks and source provenance. A rejected
  construction may be evaluated as a labeled diagnostic; it is not a promoted
  candidate. Changing project admission policy is a separate versioned decision.

## Evaluate the full text, including the failures

For every output, report independent exactness, normalized letters, flattened
words, sentence count, duplicate text/family, repeated phrases, literal
`association` frequency, phrase-unit share and reflected-word overlaps.
Report novelty relative to the frozen reference corpus, with exact copies and
adaptations separately labeled. None of these metrics certifies readability.

For this quick run, use the fixed Norvig `count_2w` table and 32 deterministic
same-word shuffles, preserving each sentence's scorer-token count. The scorer
is also related to the search objective, so it is a descriptive local-order
diagnostic, not an independent quality judge. Sentence boundaries, repetition,
exactness and reference-corpus lookups remain separate measurements. No
grammar or meaning score is inferred from these numbers.

A later independent Brown diagnostic can reuse
`experiments/score_length_stratified_readability.py`. Its predefined targets
are tokens, not letters; match actual token counts and record mismatches.
Do not reuse the older all-Brown `evaluate_polaris_debug_20260914.py` scores as
held-out evidence.

Materialized three intact prose controls per letter band, plus all eleven
controls matched to the archived exemplars: **26 distinct controls** after
deduplication. Every text reproduces its frozen September 25 surface hash.
Selection uses length and ID only. Complete contiguous sentences are retained,
never truncated spans. Their Brown document holdout does not establish
disjointness from the Norvig table. Each receives 32 shuffle comparisons.
All 189 credited Reddit palindromes are scored for descriptive comparison;
they are not assumed to be perfect prose.

If an authorized model and runtime are already available, add one blinded
0–3 grammar/meaning pass over the complete selected texts and controls, with
six hidden repeats. Reuse the existing rubric, hiding source, method and
palindrome status. Calibrate separately for short and long prose. As a small
development screen, require at least 8/9 intact controls to score at least 2
on both dimensions and at most 2/9 shuffled controls to do so in each group.
Freeze that rule before inference. Failure makes the model interpretation
inconclusive; absence of a configured model is reported, not replaced by a
claim that Brown scores measure meaning.

Final readability evidence requires people: a frozen selection spanning all
successful bands, at least three blinded readers per text, separate grammar
and whole-text meaning scores, and a one-sentence paraphrase. Do not average
sentence scores to stand in for paragraph coherence. Eric's current labels are
development feedback, not fresh confirmation.

## Decision after the quick run

Publish a matrix with one row per method/band: attempts, exact outputs,
distinct families, longest exact result, mechanical failures, model scores if
validly calibrated, and human-evidence status. Include complete representative
texts selected by a frozen rule, not just their best excerpts.

The next construction target is the first band where clear meaning fails,
even if exact length continues growing. If all candidates remain at the
grammar/meaning floor, stop the sweep: more beam width or a larger vocabulary
has not earned a scale-up. Test contextual reverse completion or stronger
relation constraints next. If the budget yields no long closures, report that
bounded failure without claiming English palindromes are impossible there.

## How the construction should improve

1. **Plan a connected event.** Carry entities, actions, objects, references and
   consequences across sentences. A paragraph can reuse an entity naturally;
   a list of unrelated palindromic blocks is not connected discourse. The
   existing term-membership semantic score does not establish these relations.
2. **Solve the two readings together.** Choose a word or phrase only after
   checking its supported reverse continuation in context. Let word breaks,
   sentence breaks and punctuation move. Judge the complete forced phrase;
   long words can have inexpensive, natural reverse readings.
3. **Make overhangs grammatical obligations.** An unmatched character suffix
   should retain its allowed word completions, verb arguments, agreement and
   unresolved references. Reject empty complete support. A low learned score
   ranks alternatives; it is not a proof of impossibility.
4. **Grow and reconstruct.** Replace paired mirrored spans jointly, recheck
   both seams and the full discourse, and keep the shorter parent if meaning
   deteriorates. Try different meanings and full reconstructions when an old
   tape traps the search. Appending individually palindromic sentences is only
   useful if the complete result gains connected meaning.
5. **Learn from joined feedback.** Preserve separate grammar and meaning
   labels, provenance and author/construction-family splits. The 183 labels
   cannot supervise particular passages until the original IDs are joined to
   verified texts. Credited Reddit text supplies examples, not automatic
   quality labels or independently generated novelty.
6. **Increase length after quality improves.** Track the best quality/length
   frontier through 1,000, then 2,000+ letters. A longer exact string alone
   does not justify wider beams, more GPUs or a larger grammar. Promote a
   mechanism only when a matched evaluation improves expressed meaning and
   preserves exactness and original construction.

These changes build on the existing shared-tape and world-state solvers. Their
small language domains already failed to produce new longer readable output;
the missing evidence is broader productive language support and actual
meaning gains. This remains a construction program, not an achieved result.

## Existing evidence this plan uses

- [Polaris search debug](../POLARIS-SEARCH-DEBUG-20260914.md): job 7618101
  closed 256/256 searches at 80–81 letters in 25 seconds; no readability claim.
- [Fixed-center evaluation](../POLARIS-CENTER-EXTENSION-20260914.md): growth
  reached 308 letters, while Brown ranked visibly poor extensions above the
  38-letter seed. This directly motivates separate reader outcomes.
- [October 8 penalty audit](../../experiments/FRAGMENT-PENALTY-AUDIT-20261008.md):
  removing targeted fragments produced replacement motifs and no validated
  human quality improvement.
- [Qwen3 audit](../../experiments/AWS-QWEN3-READABILITY-AUDIT-RESULTS-20260926.md):
  all 52 candidates scored 0/0 despite a passing narrow control screen; model
  floor effects cannot rank those candidates.
- [Scalable exact construction](../../llm_palindrome/scalable.py): optional
  one-letter fallback tests the length machinery only; never use it as a
  readable-generation arm.
- [Existing PBS example](../../tools/polaris/search_debug.pbs): historical
  one-node `debug` submission under `EVITA`. Current account, queue, executable,
  dependencies and authorization must be checked; the old staging script's
  login-node compute and destructive synchronization are not prerequisites.

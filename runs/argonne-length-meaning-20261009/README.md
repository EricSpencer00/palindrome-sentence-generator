# Argonne: length, meaning and legitimate repetition

Three Polaris jobs completed on October 9, 2026. **No longer passage passed the
whole-text meaning target.** Larger limits removed neither the limited language
coverage nor the temptation to repeat two claims. This is a negative result for
these specified methods, not a limit on meaningful English palindromes.

## Results

| Method | Completed work | Result |
|---|---|---|
| Connected document events | 126 scenes across two batches | Zero exact candidates; each scene stopped before a second event |
| Contextual extension, initial policy | 12 comparisons | Six selected outputs; longest observed exact tape 252 letters, longest whole-grammar selection 152; repeated claims |
| Larger limits, duplicate-fact rejection | 12 comparisons | Zero target selections; only 46/62-letter mechanical closures with unattached `otto` fragments |
| Fresh start, complete clauses and connected facts | 12 comparisons | Zero admitted candidates; longest unfinished frontier 76 letters |
| Qwen3-8B proposals and verifier repairs | 48 responses, including 24 feedback retries | 44 parsed non-palindromes; four malformed responses; zero exact outputs |
| Blinded machine reading | All 12 distinct exact tapes from the search outputs | Zero at 3/3 on both grammar and meaning; six at least 2/3 on both |

Lengths count lowercase ASCII letters; punctuation, case, spaces and digits
do not contribute. Unsupported alphabetic characters are rejected rather than
silently removed. Mechanical closures, complete grammar and reader acceptance
are different results. The 252-letter output is not a meaningful-length record.
Neither the 46-letter nor the 62-letter closure adds a new expressed event.

The [combined independent audit](audit/summary.json) checks original surfaces
with a raw two-pointer scan and a separately normalized reversal. It retains
all observed sources, punctuation variants, input hashes and reference matches.
It contains 12 distinct exact tapes and 26 distinct nonexact tapes. Its four
`proposal_parse_error` entries are retained model format failures, not auditor
disagreements. There are no source-length/hash or exactness-check disagreements.

## What changed

Repeated words can preserve a person, object or useful relationship. Predicate
reuse is checked against realized fact bindings, not a blanket word count.
Paraphrases of the same assertion are rejected. The fresh-start trial permits
a question followed by an assertion about the same event, while rejecting a
repeat of the same question or assertion. Referential and function phrases
are exempt from the predicate-span check.

Different bindings are only a structural precheck. Name-swapped lists can still
have poor meaning; these records are not promoted by an entity graph or a model's
description of its own plot. Grammar IDs also do not constitute a general
English coreference resolver. Natural uses outside this finite language remain
uncovered.

The follow-up raised the typed beam from 96 to 384, the length ceiling from
1,000 to 2,500 letters, and the step ceiling to 1,500. The event-state ceiling
rose from 200,000 to 1,000,000 per scene. It also added duplicate-fact rejection;
the change therefore does not isolate the causal effect of beam width.

The fresh-start method removes the 38/54-letter frozen center. It uses complete
clause frontiers with independently matching word boundaries, a 96-letter
overhang ceiling, no global word-use cap and no incumbent length stop. The
historical `beam_search` defaults remain unchanged; the new options are explicit.
Question/statement distinctions and a connected fact graph are scoped
experimental constraints, not a universal definition of meaningful prose.

## Failure boundaries and local minima

Each event scene explored 64 distinct states and exhausted its available
frontier. Raising its state ceiling did not introduce a second event. Its
saved diagnostics show the conflicting letters at the next event boundary.
The repeated-name/role vocabulary and fixed document-event realizations are
too restrictive for this task; additional beam width cannot add missing forms.

The initial contextual method generated longer text by repeating “an aide rips
nine memos” and “some men inspire Diana.” Removing repeated facts eliminated
its target outputs. This is the observed degeneration, rather than a claim that
all repeated words are bad.

The fresh-start runs each considered 1,179 candidate states, of which 240 had
supported grammar frontiers. In the 30-letter band, each checked 26 distinct
exact closures: 20 lacked a complete grammar parse, four repeated facts and
two had disconnected fact graphs. No admitted closure reached any tested band.
No deadline was hit. Its finite beam and per-parent choices remain limits;
`engine_returned` is not an UNSAT proof. Rejected-closure reasons are counted;
the wrapper saves admitted candidates and unfinished frontier evidence, not
every rejected surface.

The first model batch requested exact lengths. The second allowed ranges
(120–239, 240–479, 480–959 and longer reconstruction ranges), preserving a
minimum and forbidding filler. Both used one immediate verifier-feedback retry.
Neither produced an exact passage. Looser target lengths alone did not fix this
proposal-plus-filter baseline. It is not a constrained decoder or a test of
every language model.

The next useful search change is broader, boundary-supported English
realizations with maintained referents and useful consequences. Propose them
where the saved grammar/letter frontiers conflict, admit only supported exact
paths, and evaluate whole passages. Another increase to the same exhausted
templates is not evidence of progress.

## Reading evaluation

The evaluator was a separate blinded `gpt-6.1-sol` agent at `xhigh`, not
Argonne-hosted inference and not a human study. It first rated 18 frozen
calibration texts. The preset narrow criterion required at least 80% of intact
texts, and at most 20% of same-word shuffled texts, to score at least 2/3 on both
dimensions. It passed: 8/9 intact, 0/9 shuffled. This small calibration does not
establish general judge accuracy.

The initial ten exact tapes and the two later mechanical closures were rated
in separate packets without source claims or scores. All 12 scored meaning 2;
none scored meaning 3. Grammar ranged from 1 to 3. Original rating IDs are joined
to normalized-tape IDs in [matched-machine-ratings.json](matched-machine-ratings.json).
The goal still requires fresh human evaluation of complete passages; no human
readability result is claimed.

## Compute and provenance

| Polaris job | Queue/start UTC | Scheduler wall time | Exit |
|---|---|---|---|
| 7731518 | 17:44:46 / 17:46:38 | 5m 15s | 0 |
| 7731535 | 17:56:39 / 18:07:10 | 6m 51s | 0 |
| 7731549 | 18:10:38 / 18:26:24 | 1s | 0 |

All used one node under `EVITA`, `debug`. The first two reserved 32 CPU cores
and four GPUs; inference used one A100 40 GB. The third used three concurrent
CPU processes and no GPU inference, with a dependency on the second job.
Scheduler wall time totals 12m 7s; queue and scheduler overhead are separate.
The third's roughly 0.188 summed search seconds explain its short execution.
Its long wait does not represent more search.

The model is [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B), frozen at
`b968826d9c46dd6066d109eabc6255188de91218`. The recorded runtime is Python
3.13.15, PyTorch 2.14.0, Transformers 5.10.4 and CUDA 13.0; inference used BF16,
`enable_thinking=False`, temperature 0.7, top-p 0.8 and top-k 20. It downloaded
anonymous public weights once and reused that task-owned cache. No model was
trained or fine-tuned.

The demonstration-conditioned arm keeps credit for the 137-letter Reddit title
posted by [u/catinore](https://www.reddit.com/user/catinore/) at
[its original post](https://www.reddit.com/r/palindromes/comments/1uphsc9/put_it_up_is_a_noose_so_meg_never_idle_i_yet_saw/).
Credit identifies the poster, not independently established original authorship.
Conditioning on or adapting a source is not independent discovery. No model
output from this arm passed exactness.

Only public code, lexical counts, credited posts and existing public research
constructions went to Argonne. The private 183-row reader export stayed local.
Their unresolved passage mappings remain unresolved; no labels were invented
or used for training. No AWS resource or paid inference endpoint was used.

## Reproduction and files

- [summary.json](summary.json): outcomes and scheduler times.
- [verification.json](verification.json): source hashes and successful exits.
- `initial/`, `expanded/`, `free/`: untouched returned results, logs, scheduler records and manifests.
- `source-snapshots/initial/` and `source-snapshots/expanded/`: historical runner/library versions matching the first two manifests.
- `source-snapshots/public-inputs/`: the three pinned legacy exemplar inputs, available even in a fresh Git checkout.
- `early-audit/`, `expanded-audit/`, `free-audit/`, `audit/`: independent validation, blind packets and separate keys.
- `calibration-ratings.json`, `early-meaning-ratings.json`, `expanded-meaning-ratings.json`: unmodified blinded ratings.

The current tools combine fresh-start trials into the expanded batch for later
runs. This session instead executed the fresh-start PBS script separately
because the preceding job had already started. Use the saved version matching
each manifest to replay that historical batch. Restore its three exemplar files
from `source-snapshots/public-inputs/runs/` if the original ignored paths are absent;
the current model loader also supports this scoped fallback and refuses changed
present originals.

The standalone CPU replay is `tools/polaris/free_typed_search.pbs`. The current
combined CPU/GPU replay is `tools/polaris/meaning_search_batch.pbs`; supplying
`PAL_PUBLIC_MODEL_CACHE` reuses an existing verified public cache, otherwise the
runner creates a fresh cache. Both require task-scoped staging on Eagle and the
existing allocation. No remote scratch path is a durable result location.

The exactness audit can be repeated per returned `results/` directory with
`experiments/audit_argonne_meaning_batch_20261009.py --results DIR --out NEW_DIR`.
The combined audit uses the same `build_audit` function on those three raw-file
collections, prefixed with their phase names. Original bytes and every source
hash remain available. [The reuse policy](../../docs/research/2026-10-09-meaning-and-reuse.md)
and [30-line tuning explanation](../../docs/TUNING-30-LINES.md) describe the intent.

Validation: 108 relevant tests, independent full-tape checks, all 69 + 69 + 8
source hashes verified, all three scheduler exits successful. Raised limits,
fresh starts and legitimate reuse are implemented; the meaningful-length
milestone remains unachieved.

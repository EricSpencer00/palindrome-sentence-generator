# Plan for a stronger NAACL result

Date: 2026-09-25. Status: proposed plan only. No implementation, experiment, recruitment, or spending is authorized by this document. It supersedes older experiment proposals for the purpose of deciding the next paper study; it does not change existing construction artifacts.

## The result the paper needs

**At a fixed generation budget, can exact palindrome search produce more distinct sentences that people find grammatical and understandable than a faithful published generator and strong existing alternatives?**

The current resource-sensitivity experiment is useful pilot evidence. It establishes that larger resource bundles changed exact-output yield from 0/120 to 66/120 on the frozen tasks. It does not establish improvement over published work or improvement in human readability. The 66 outputs include only 52 distinct texts, 36 share “one poem association,” and there are no human ratings. Moving that result into the pilot/appendix makes room for the actual contribution. See [the experiment report](../experiments/DIRECTIONAL-SEARCH-RESOURCE-SENSITIVITY-20260925.md) and [the current manuscript](naacl2027.tex).

The proposed contribution has three connected parts: a well-defined comparison with prior methods, a human evaluation of useful output yield, and an ablation explaining a specific search improvement. A new architecture is optional. Evidence of a substantial, reproducible advance is necessary. This plan cannot promise a reviewer score of 5.

## Evidence shaping the decision

| Evidence | Consequence for this study |
| --- | --- |
| [Papadopoulos et al., IJCAI 2015](https://www.ijcai.org/Proceedings/15/Papers/353.pdf) already generated very long palindromes from n-gram corpora and discussed their weak meaning. | Length and exact validity are established capabilities. Reproduce this published algorithm as the principal external palindrome baseline. |
| [Nichol, 2025](https://blog.aqnichol.com/2025/06/24/finding-palindromes-with-language-models/) describes a dedicated palindrome language model. This is an author writeup, not a peer-reviewed evaluation. | Include it in the novelty audit. Use an executable baseline only if code/checkpoints can be obtained promptly; availability is currently unverified. |
| [Bonlarron and Régin, 2024](https://www.ijcai.org/proceedings/2024/841), and [Bonlarron et al., 2025](https://www.ijcai.org/proceedings/2025/1115), combine language models and constraint reasoning. Their evaluated tasks differ from ours. | “Combine a language model with constraints” is insufficient novelty. State the precise difference, or frame the contribution as an empirical study. |
| The [repository README](../README.md) already describes LM reranking and in-loop scoring. The superseded [research strategy](../docs/RESEARCH-STRATEGY.md) records implementation problems that invalidate several historical conclusions. | Establish the strongest corrected pre-existing project method. Do not introduce familiar features as new contributions or use broken historical code as a weak comparator. |
| [Howcroft et al., INLG 2020](https://aclanthology.org/2020.inlg-1.23/) documents inconsistent human-evaluation practices in NLG. | Publish explicit criteria, instructions, sampling, rater qualifications, exclusions, and agreement. |

## 1. Establish credible comparators: September 26–27

Freeze a provenance table distinguishing original released code, faithful reimplementations, project-local methods, and new changes. Record source URLs, dates, versions, adaptations, and unavailable artifacts.

Use four principal arms:

1. **Published n-gram graph generator:** reproduce the 2015 algorithm. First seek author code; if unavailable, label our implementation explicitly. Check tiny exhaustive cases and published examples before claiming a faithful reproduction. A paper citation alone does not validate our implementation.
2. **Strongest pre-existing project method:** select on development data among corrected native search, LM reranking, and in-loop LM scoring. Include any previously available grammar support. Freeze its revision before introducing the proposed change.
3. **A capable language model with exact validation and repair:** freeze the model identifier, prompt, decoding settings, retry limit, and total token/cost allowance. Save every attempt, including failures. This is a newly specified operational baseline, not a published palindrome system.
4. **The proposed search change**, described below, built on arm 2 with all unrelated settings held fixed.

Timebox investigation of Nichol's executable artifacts to this stage. Do not train a replacement model from scratch merely to fill a comparison row. Document omission if the artifacts remain unavailable.

For the published method, preserve its native behavior and document any adaptation needed for common inputs. Compare on a shared, sourced vocabulary/corpus where possible, and report its native-resource result separately if harmonization changes its behavior substantially. Charge and report preprocessing costs separately. Project-specific shortcut exclusions must not silently disqualify baseline outputs from the primary validity measure.

**Gate:** proceed only when the external implementation passes conformance checks and a strong project comparator is frozen. Otherwise repair the comparison or narrow the paper's claim before purchasing the main study.

## 2. Test one substantive mechanism: September 28–30

The working hypothesis is that search fails linguistically because it commits words on two mirrored frontiers without adequately testing whether their unfinished grammatical structures can meet.

The candidate intervention is **joint grammatical completion checking during exact search**: track unfinished clause/phrase requirements at both word frontiers, check their compatibility before expanding expensive continuations, and retain the existing exact character invariant. The first action is an audit of existing grammar and semantic-state implementations. Reuse the best existing mechanism; do not implement or claim the same idea twice. If the candidate is already present, identify a specific missing compatibility check or choose an empirical contribution instead.

Compare the candidate with the same scorer, vocabulary, beam allocation, and budget without that check, plus a version that applies the same language check only to finished candidates. This separates the value of early pruning from the value of a better final filter. Grammar checks do not establish meaning; human judgments remain the outcome.

Use development trials and a small blinded human pilot. Allow at most two clearly specified candidate variants over two development days. Freeze the winning variant, output-selection rule, rating instructions, and analysis plan before the held-out experiment. Stop this method direction if its apparent advantage disappears against the strongest corrected existing method or merely suppresses nearly all outputs.

## 3. Run a fixed comparison with human outcomes: October 1–6

Initial design: two independently sourced language resources × three normalized-letter bands (20–39, 40–59, 60–89) × 20 held-out random seeds = **120 generation trials per arm**. Four principal arms plus the post-hoc-filter ablation give up to 600 returned texts before deduplication. Development seeds and examples are disjoint. Freeze exact resource identities, versions, licenses, and preprocessing before the test.

These trials sample generation behavior; different seeds are not different linguistic tasks. A deterministic baseline must not be rerun identically 20 times and counted as independent evidence. Its native enumeration/sampling interface and effective number of independent runs must be established in the pilot; if needed, revise and freeze the design before collecting test labels. Report conclusions conditional on the chosen resources and bands, not universal language-generation performance.

Use a common total budget per trial, record time-to-output, and report quality at several budget checkpoints. For code-based methods, use the same remote hardware. For API inference, report dollars, tokens, model calls, and latency rather than pretending API seconds equal local search seconds. All generation, repair, and selection costs count. Report an equal-cost comparison alongside elapsed time; preprocessing/training costs remain visible.

Return one text per trial using a frozen selection rule. Include empty, invalid, and timed-out trials in denominators. Do not inspect human scores to choose a different candidate. Hash and deduplicate texts before annotation; annotate a repeated text once and preserve its occurrence count. Report both successful-trial yield and distinct accepted texts per total budget, so recycled sentences cannot create the headline gain.

Recruit three independent fluent-English raters per unique text, blinded to method and hypothesis. Ask separately whether the text is grammatical and whether it expresses an understandable meaning without supplying an invented explanation. Use a brief paraphrase on a prespecified subset to check that apparent understanding survives a comprehension task. Randomize order and mix sourced human palindromes, ordinary prose, and deliberately disordered controls. Freeze exclusion rules from the pilot; publish agreement and rating distributions, not only majority votes. Complete the applicable human-participant ethics and consent process before recruitment.

Primary outcome: the fraction of all trials returning an exact palindrome accepted on both grammaticality and understandability by at least two of three raters. The exact palindrome check is deterministic and independent of language scoring. Catalog matches, repeated phrase families, symmetry shortcuts, and distinct output count are prespecified secondary outcomes. “Absent from our frozen catalog” must not be called proof of global novelty or absence from model training data.

The 120-trial count is an initial design, not a power guarantee. Use pilot yield, duplication, and rating variation to simulate precision and fix the final affordable sample size before test generation/labels. Estimate the paired difference against the strongest comparator chosen on development data, respecting repeated texts and actual independent generation units. Show uncertainty per resource and length band. Keep comparisons with other arms secondary. Do not extend the study until a significance threshold is crossed.

## 4. Define what counts as a worthwhile result

Proposed planning gates, not thresholds taken from the literature:

- At least **15 percentage points more human-accepted trials** than the strongest development-selected baseline, with a 95% interval for the difference excluding zero.
- At least **30 distinct human-accepted outputs** from the proposed method; report all of them, not a handpicked gallery.
- A positive estimated gain on each resource; show the uncertainty rather than hiding a failed resource inside the pooled average.
- A gain that survives reporting distinct outputs and dominant phrase families separately, with the controlled ablation supporting the claimed mechanism.
- The advantage holds at a comparable generation cost and is not explained solely by increased compute or a weaker baseline.

If these gates fail, do not manufacture a positive method story. A negative/evaluation paper is justified only if it establishes a substantial, reusable finding—for example, a validated failure of widely used language scores to track human judgments across methods. Otherwise defer the submission. Failure of the experiment is allowed; changing the claimed result to match convenient metrics is not.

## 5. Readability audit, manuscript, and QA: October 7–10

Keep GPT-2 and Brown as frozen likelihood diagnostics. Compare them against the collected human labels using prespecified association and error analyses; examine length, repetition, and catalog matches as confounds. Any scorer used for generation must be identified as such. DeepSeek ratings are a secondary exploratory check, with stored calls and a stability assessment if retained. None is a static verifier of meaning. Traditional school-grade formulas are inappropriate as the primary quality measure for these short, unusual texts.

Rewrite the paper around the question, matched baselines, human outcome, cost, and mechanism. Include randomly sampled outputs and representative failures. Move the existing resource study to the development history/appendix and identify its adaptive role. Provide a claim-to-artifact index linking every reported number to immutable inputs, code/model versions, raw outputs, analysis, and anonymized ratings. Recomputing saved results must be distinguished from rerunning a hosted model, whose output may change.

Final QA should include an independent skeptical research reviewer and an independent prose reviewer. Require them to identify unsupported novelty, comparison advantages, language claims, and inconsistencies between tables and artifacts. Fix findings before submission, then perform artifact reproduction and PDF checks. Their ratings remain advice, not a predicted acceptance result.

## Schedule and budget

The [NAACL 2027 call](https://2027.naacl.org/calls/main_conference_papers/) lists October 12, 2026, 23:59 AoE for ARR submission and reviewer registration. Reserve October 11–12 for submission checks. Recruitment and any ethics review are critical-path dependencies; if they cannot finish in time, do not relabel model judgments as human evaluation.

Proposed compute envelope: **$100 total** ($20 development/pilot, $60 fixed experiment, $20 reserve), with **aggregate AWS spend below $10/hour**, job timeouts, and automatic teardown. Prefer an available authorized remote host; keep substantive inference off the coordinating Mac. Price the chosen resources before launch and shrink the design before freezing it if the envelope is inadequate. This is a planning cap, not a measured cost estimate or an instruction to launch jobs.

Human annotation needs its own budget. At 600 unique texts × three raters × 45–90 seconds per rating, annotation requires roughly 22.5–45 worker-hours. At a proposed $18/hour, that is $405–810 before pilot work, controls, recruitment, or platform charges. Plan **$600–1,200**, then replace this estimate with measured pilot timing and an actual recruitment quote. No participant payments or recruitment have been initiated.

Rejected uses of the budget: further length records, another larger-beam-only headline, large-model training, many judge prompts without humans, or polishing prose before establishing a defensible result. The next executable step, if separately resumed, is the baseline/provenance audit. This planning turn stops here.

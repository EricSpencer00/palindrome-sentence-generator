# 24-hour plan for a stronger NAACL draft

Revised: 2026-09-25. Status: proposed plan only; execution remains stopped. The user requires a complete professor-review packet within 24 hours, followed by at least seven full days for professor review. This revision replaces the previous multiweek execution schedule. It does not launch implementation, experiments, recruitment, or spending.

H0 means the start of a separately resumed execution run. If work starts September 25, handoff is September 26 at the same time of day; the professor's review window lasts until October 3 at that time. Do not consume that review window by moving the research deadline. No professor message or calendar invitation has been sent.

## The result the paper needs

**At a fixed generation budget, can exact palindrome search produce more distinct sentences that people find grammatical and understandable than a faithful published generator and strong existing alternatives?**

The current resource-sensitivity experiment is useful pilot evidence. It establishes that larger resource bundles changed exact-output yield from 0/120 to 66/120 on the frozen tasks. It does not establish improvement over published work or improvement in human readability. The 66 outputs include only 52 distinct texts, 36 share “one poem association,” and there are no human ratings. Moving that result into the pilot/appendix makes room for the actual contribution. See [the experiment report](../experiments/DIRECTIONAL-SEARCH-RESOURCE-SENSITIVITY-20260925.md) and [the current manuscript](naacl2027.tex).

The proposed contribution has three connected parts: a well-defined comparison with prior methods, a human evaluation of useful output yield, and an ablation explaining a specific search improvement. A new architecture is optional. Evidence of a substantial, reproducible advance is necessary. This plan cannot promise a reviewer score of 5.

## Evidence shaping the decision

| Evidence | Consequence for this study |
| --- | --- |
| [Papadopoulos et al., IJCAI 2015](https://www.ijcai.org/Proceedings/15/Papers/353.pdf) already generated very long palindromes from n-gram corpora and discussed their weak meaning. | Length and exact validity are established capabilities. Prefer this comparator if a faithful executable version passes the early checks; do not promise an overnight reproduction. |
| [Norvig's palindrome generator](https://www.norvig.com/palindrome.html) has [public source](https://github.com/norvig/pytudes/blob/main/py/pal2.py). | A real pre-existing external implementation is available for an immediate baseline audit. It is historical, not a current state-of-the-art claim; its native objective and input constraints require checking. |
| [Nichol, 2025](https://blog.aqnichol.com/2025/06/24/finding-palindromes-with-language-models/) describes a dedicated palindrome language model. This is an author writeup, not a peer-reviewed evaluation. | Include it in the novelty audit. Use an executable baseline only if code/checkpoints can be obtained promptly; availability is currently unverified. |
| [Bonlarron and Régin, 2024](https://www.ijcai.org/proceedings/2024/841), and [Bonlarron et al., 2025](https://www.ijcai.org/proceedings/2025/1115), combine language models and constraint reasoning. Their evaluated tasks differ from ours. | “Combine a language model with constraints” is insufficient novelty. State the precise difference, or frame the contribution as an empirical study. |
| The [repository README](../README.md) already describes LM reranking and in-loop scoring. The superseded [research strategy](../docs/RESEARCH-STRATEGY.md) records implementation problems that invalidate several historical conclusions. | Establish the strongest corrected pre-existing project method. Do not introduce familiar features as new contributions or use broken historical code as a weak comparator. |
| [Howcroft et al., INLG 2020](https://aclanthology.org/2020.inlg-1.23/) documents inconsistent human-evaluation practices in NLG. | Publish explicit criteria, instructions, sampling, rater qualifications, exclusions, and agreement. |

## The 24-hour schedule

Run the experimental, annotation, and manuscript preparation work in parallel once execution is resumed. Codex owns implementation, tests, and artifact QA; a research subagent audits comparison fairness and a separate reviewer critiques the final paper. Those are future assignments, not agents launched by this planning revision.

| Time from start | Work | Deliverable and cutoff |
| --- | --- | --- |
| H0–2 | Audit existing implementations and external source code; check remote capacity and prices. In parallel, establish availability of raters and an appropriate consent/recruitment process. | Source/version ledger, candidate methods, and a feasible annotation schedule. Identify a human-evaluation blocker immediately. |
| H2–4 | Repair only blocking defects, check exactness and baseline conformance, run small development trials and a rating pilot. | By H4 freeze methods, resources, sample size, prompts, selection rules, rating rubric, and analysis. No new architecture project. |
| H4–10 | Execute the fixed experiment on remote workers. Build anonymous annotation batches as each frozen batch completes. Draft methods and related work alongside the runs. | Complete outputs, failures, timings, cost records, and a manuscript with methods already written. |
| H6–16 | Obtain three independent ratings per unique text, with randomized order and prespecified controls. | A locked rating dataset and recorded exclusions/agreement. This overlaps generation; it does not start after all analysis is finished. |
| H16–20 | Analyze the primary outcome, duplication, ablation, uncertainty, and score calibration; write results and limitations. | Complete draft with numbers linked to saved evidence. Freeze the result tables at H20. |
| H20–23 | Independent scientific and prose review; reproduce tables from saved artifacts; run focused implementation checks and inspect the PDF. | Corrected manuscript, evidence archive, and explicit remaining limitations. |
| H23–24 | Package the professor handoff. | PDF, one-page contribution/results summary, baseline provenance, reproducibility instructions, and three focused review questions. |

## Scope that fits the deadline

Use **four arms**, with the exact choices locked by H4:

1. **The strongest working pre-existing project method**, including existing language/grammar support. Select on development data and preserve its source snapshot. Studying it properly can be a contribution; merely renaming it cannot.
2. **One controlled ablation** of that method: preferably in-loop language checking versus the same check applied after generation. Hold the scorer, vocabulary, and cost allowance constant. If that comparison is not supported by working code at H4, use another existing, clearly isolated component and rewrite the hypothesis before freezing.
3. **A real external generator.** Start with the available Norvig source. Prefer the 2015 method only if an executable implementation passes conformance checks by H4. Record original source, all ports/adaptations, native constraints, and reproducible examples. Norvig is a historical reference, not proof of modern state-of-the-art superiority. If its native task cannot be fairly adapted, report it separately and explicitly narrow the main comparison.
4. **A capable language model with exact validation and repair**, using a fixed model, prompt, retry limit, and total budget. This arm is newly specified for this study; do not call it a published baseline. Preserve every attempted output.

Keep the 2015 and Nichol work in the related-work/novelty analysis even if executable artifacts are unavailable. Limit artifact discovery to the first two hours. Do not train a new model, invent a new grammar engine, or start an unvalidated reimplementation after H4. The deliverable is a rigorous empirical result with existing methods, with any methodological novelty claimed only when the code history and literature support it.

Initial main-study design: **two independently sourced language resources × three normalized-letter bands (20–39, 40–59, 60–89) × 15 held-out seeds = 90 trials per arm, 360 total**. Freeze versions, licenses, normalization, input resources, and disjoint development/test seeds. Seeds measure generation variability, not new semantic tasks. Identical deterministic reruns cannot count as independent samples; resolve each baseline's sampling/enumeration unit before H4.

At a 180-second trial ceiling, 360 trials require at most 18 worker-hours of generation, or about three hours across six isolated workers before setup/overhead. The H4–10 window includes reserve. This is scheduling arithmetic, not a measured throughput promise: the pilot must confirm hardware capacity, API limits, and full end-to-end timing. Fix any necessary size/budget reduction at H4, before held-out results are observed. Use the pilot to assess expected precision; 90 trials per arm is not a power guarantee.

Use identical per-worker resources for code-based methods. Report API dollars, model calls, tokens, and elapsed time separately, with equal-cost results wherever the saved budget traces permit. Include selection/repair costs and report preprocessing separately. Do not disadvantage the external method with project-specific anti-shortcut rules. Return one candidate per trial under a frozen rule; invalid, empty, and timed-out trials remain in the denominator.

## Human evidence is on the critical path

Target **six to eight available raters**, assigning each unique output to three different fluent-English readers who are blind to method and hypothesis. Deduplicate before annotation and preserve occurrence counts. Rate grammaticality and understandable meaning separately; include a brief paraphrase on a prespecified subset. Mix sourced human palindromes, ordinary prose, and disordered controls. Publish instructions, exclusions, agreement, and rating distributions.

At 360 unique outputs, three ratings each, and an estimated 45–90 seconds per rating, the main study needs **13.5–27 worker-hours**. Six to eight raters can divide that into roughly two to five hours each, plus the pilot and controls. Confirm actual availability and timing at H0–4; parallel compute does not solve missing human availability.

Primary outcome: the fraction of all trials producing an exact palindrome judged both grammatical and understandable by at least two of three raters. Report distinct accepted texts per budget alongside it. Repeated phrase families, catalog matches, and structural shortcuts are separate analyses. Absence from a frozen catalog does not prove global novelty or absence from training data.

If raters or the required institutional process cannot be arranged promptly, preserve the H24 handoff with an explicitly incomplete human-evaluation section and the ready-to-run annotation packet. Do not present model ratings as human evidence. If labels are incomplete at H16, report their coverage and restrict conclusions accordingly; missing labels are unknown, not negative judgments. The professor receives the completed work and the exact limitation, rather than waiting beyond the deadline.

## Success criteria and analysis

Retain the proposed substantive targets: **at least a 15-percentage-point improvement** over the strongest development-selected comparator, a 95% interval excluding zero, **at least 30 distinct human-accepted outputs**, and positive estimated gains on both resources. These are planning gates, not existing results or literature-defined thresholds. Require the advantage to survive cost and duplication analyses and the ablation to support its explanation.

Use paired comparisons where the input/run units truly match, respect shared texts and generation dependencies, and show uncertainty by resource and band. Fix the estimator and comparisons at H4. Do not increase the sample until a desired score or significance level appears. If the gates fail, report that finding and narrow the claims; the professor's review can judge whether the empirical contribution remains sufficient.

GPT-2 and Brown remain frozen likelihood diagnostics. Relate them to the human judgments with prespecified association/error analyses, checking length and repetition. Identify any overlap between selection and evaluation models. Reuse saved DeepSeek evidence as exploratory; additional judge-prompt sweeps are outside the 24-hour scope. Exactness is mechanically verifiable; meaning requires the human evidence. School-grade readability formulas are not the main outcome.

## The professor's seven-day review window

**H24 handoff: September 26 if execution starts September 25. Review: September 26–October 3, ending no earlier than the handoff time.** Give the professor a stable, versioned draft and ask:

1. Does the evidence support a substantial contribution for NAACL?
2. Are the baseline adaptations, human evaluation, and uncertainty defensible?
3. Which claims or explanations need narrowing before submission?

During the seven days, keep the handed-off draft identifiable and collect corrections separately. Do not require a faster response or plan another large experiment as a prerequisite to their review. After October 3, incorporate feedback and run only necessary targeted follow-ups. Reserve October 11–12 for final checks. The [NAACL 2027 call](https://2027.naacl.org/calls/main_conference_papers/) gives October 12, 2026, 23:59 AoE for ARR submission and reviewer registration.

The packet includes the PDF, all output examples and failures, anonymized ratings if collected, source/version manifest, analysis commands, and a claim-to-artifact index. Distinguish reproducing tables from saved outputs from rerunning a hosted model. Keep the current resource-sensitivity result as pilot evidence. The one-page summary should state what improved, by how much, compared with what, at what cost, and what remains unsupported.

## Budget and stop rules

Proposed compute cap remains **$100 total**, with **aggregate AWS below $10/hour**, bounded jobs, and teardown. Use authorized remote capacity for inference; the Mac remains the coordinator. Price the actual configuration before launch. Proposed annotation envelope is **$400–800**, based on $18/hour plus pilot work, controls, and platform fees; replace it with pilot timing and a real recruitment quote before spending.

The hard stops are H4 for scope, H10 for the planned main runs, H16 for the primary annotation cutoff, H20 for result tables, and H24 for handoff. A failed experiment changes the claims, not the clock. No length-record campaign, larger-beam-only headline, model training, or speculative feature work belongs on this schedule. This revision changes only the plan; execution remains stopped.

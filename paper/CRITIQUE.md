# Critique of the current palindrome paper

Historical review of the Markdown manuscript `eric_evidence_release_draft.md`
from commit `e87bbb7`, completed 10 September 2026. The user subsequently requested
a rewrite in the earlier Eric draft's voice and a LaTeX version. The current
manuscript is [naacl2027.tex](naacl2027.tex); the findings below record the review
before that rewrite. Several prose, methods, and presentation issues have since
been addressed, while the experimental limitations remain.

My assessment: the paper supports a useful, narrow engineering observation, but the current draft does not yet make a convincing research contribution. The main weaknesses are its treatment of closely related work, the single bounded comparison, and the untested value of the additional candidates. The numerical result itself survives inspection.

## What holds up

The existing saved-artifact audit passes. Terminal filtering retained 20,989 distinct token pairs and incremental filtering retained 86,511, a ratio of 4.1217. The 90,937-letter artifact also passes reversal, dictionary-membership, phrase-uniqueness, and repetition checks; it exceeds the identically parsed 90,439-letter Norvig reference by 498 letters. These are defensible artifact claims.

I additionally checked all 107,500 saved pair rows against the frozen vocabulary and an independently implemented tag-pattern checker. That checker uses bitsets over pattern positions rather than the production `SentencePlan` implementation. Every row passes exact mirror equality, token uniqueness, length bounds, and complete pattern admission. This verifies saved outputs, not the original timing, absence of missed solutions, or readable English.

The paper's explicit separation of structural admission from readable prose is a strength. Keep the awkward accepted example: it gives readers a more accurate understanding of the filter than the term “sentence pattern” alone.

## 1. The novelty case needs a direct comparison with existing palindrome work

Location: abstract, Sections 1–2, references.

The abstract effectively asks whether impossible partial syntactic states can be rejected early. That is too familiar a question to carry the contribution by itself. More specifically, Papadopoulos et al. already describe syntactic-pattern graph conjunction in Section 2.1 and apply it to palindrome generation in Section 3.4. Their paper also discusses pruning incompatible graph states. Listing it in the bibliography without explaining the relationship leaves the central contribution exposed. [Primary paper, IJCAI 2015](https://www.ijcai.org/Proceedings/15/Papers/353.pdf).

The defensible distinction to investigate is the particular combination of center-out overhang search, separate sentence-pattern constraints on the two halves, ambiguous observed tags, repetition restrictions, and finite-budget candidate yield. The draft needs to explain which part differs from earlier methods and why that difference matters. It cannot assume that implementing an established constraint-propagation idea in a different representation is sufficient novelty.

Add a focused related-work paragraph and a comparison on a common, explicitly defined feasible set, or present this as a carefully bounded implementation study. The neural-decoding references currently do little argumentative work because the main experiment contains no neural decoder.

## 2. One run supports the recorded yield difference, not a reliable general advantage

Location: Section 2, especially the experiment paragraph and final interpretation.

The manuscript correctly discloses shared stopping caps, fixed terminal-first arm order, changed random draws, and missing process-level traces. Those qualifications prevent overclaiming, but they do not replace experimental evidence. The 32 processes partition one run; they are not 32 independent replicated comparisons. Global deduplication further complicates treating their outputs as independent observations.

The result needs repeated runs with independently varied seeds, balanced or randomized arm order, saved stopping reasons, and yield over elapsed time. Log generated children, time spent checking patterns, memory use, and popped states separately. A second comparison under a fixed state budget can clarify the mechanism, but equal popped-state counts would still not imply equal work because the incremental arm checks children that never enter the stack.

There is no honest confidence interval for the fourfold multiplier available from the surviving aggregate alone. Retain the present descriptive ratio and avoid promoting it into a general performance claim until those trials exist.

## 3. More token pairs do not establish a proportionate increase in useful variety

Location: Sections 2–3 and the claim that structural pairs have downstream use.

I computed the following directly from the saved candidate lists. A junction family is the ordered pair `(last left word, first right word)`, matching the repository's existing family definition. It is a descriptive grouping, not a claim of statistical independence or semantic equivalence.

| Saved-output measure | Terminal | Incremental |
| --- | ---: | ---: |
| Distinct token pairs | 20,989 | 86,511 |
| Distinct normalized letter strings | 19,479 | 82,056 |
| Distinct junction families | 20 | 30 |
| Outputs at the 44-letter ceiling | 83.45% | 73.49% |

The string counts show that alternative segmentation does not explain away the gain. Nevertheless, the family count grows only 1.5-fold. One incremental family, `rates / set`, accounts for 10,000 pairs. This is a reason to measure concentration, not proof that those pairs are interchangeable or useless.

Report these distributions, plus overlap between arms (16,766 token pairs), and examine whether the extra pairs survive the composition constraints. A fixed-bank composition comparison could measure accepted components, lexical reuse, and achievable length without claiming improved readability. Human evaluation becomes necessary if the paper claims a benefit in meaning; it is not automatically the next experiment for a paper whose stated endpoint is structural search yield.

Reproduction: `python3 paper/critique_evidence.py`. Input hashes and full counts are in [critique-evidence.json](critique-evidence.json).

## 4. The introductory construction teaches the opposite growth direction

Location: Section 1, the `step on / pets` example and the paragraph after it.

The text says the search grows toward a meeting point and illustrates prepending `no` to the right piece. It then correctly says the implemented pair search prepends to the left and appends to the right. These are different growth conventions. Since the suffix/prefix pruning argument depends on direction, this is a methods error rather than merely a stylistic inconsistency.

Use a center-out trace consistent with `llm_palindrome/centerout.py` and `llm_palindrome/exhaustive.py`: start with `on`, append `no` on the right, prepend `step` on the left, then append `pets` on the right. Distinguish this letter-mechanics illustration from an accepted pair under the three-word minimum.

Add a short invariant stating that each existing left sequence remains a suffix of its final half and each right sequence remains a prefix. Follow it with a no-false-pruning argument relative to the frozen pattern language and an exhaustive small-instance comparison of terminal and incremental admission. The current tests exercise the pattern predicates and generic pruning hook, but do not establish that full integration equivalence.

## 5. Essential experimental definitions are missing from the main account

Location: Sections 2–3 and Reproduction.

The reported structural run is bounded to 20–44 normalized letters, 18 total words, and an overhang of at most 16. Its vocabulary file contains 28,402 entries, of which 21,073 are Brown-known and used by both arms. These settings are visible in the code and frozen inputs but omitted from the main experiment description. The very high concentration at 44 letters makes the missing length bounds especially consequential.

The composition paragraph refers to a “stated exception set,” but the manuscript does not actually enumerate that set or give its numerical repetition caps. Its composition exceptions also differ from the long-artifact auditor's exceptions. Name the tokenizer, exact sets, and cap values for each experiment separately; otherwise “maximum nonexception-word count” is not fully interpretable from the paper.

The original release was an evidence collection, not an executable reproduction package: it omitted the imported search modules and frozen Brown/vocabulary payloads required by its verification command. The cleanup now states the repository requirement explicitly. A stronger release would provide a precise checkout/environment and an input manifest covering all those dependencies. Hashing available files cannot establish the missing historical experiment revision.

Also reconcile the historical source audit's “both passing models” statement with the current draft's one passing model. The ledger's historical seam row points to the source audit, which describes missing files but does not give the promised historical totals; point to the actual retained report as well.

## 6. The paper's structure spends too much space explaining its own revision

Location: abstract, Section 3, Section 4, conclusion, and Evidence release.

A whitespace count gives roughly 1,618 words before the evidence release and 1,714 after it, including references. The ledger is useful supplementary documentation, but its size and the “what changed” section make the manuscript read partly like an internal correction report. A new reader has no reason to care about withdrawn claims before understanding the contribution being retained.

Keep the main paper centered on the structural method, related work, experiment, and limitations. Move the historical correction narrative into supporting documentation. Reduce the composition algebra and the long Norvig artifact to their actual supporting roles; neither validates the structural filter. State the absence of a readability result once clearly, then use the recovered space for the missing method and evaluation details.

The conclusion currently calls independent human assessment the next decisive evidence. That follows the broader project ambition, but not this paper's narrowed claim. Its immediate evidence needs are a defensible comparison with prior methods, reproducible repeated yield measurements, and a test of what the extra candidates contribute.

## Revision order

1. Correct the growth example and specify the complete experimental setting.
2. State the precise difference from prior palindrome constraint methods.
3. Add the already available diversity and concentration measurements.
4. Run replicated comparisons and test whether additional candidates improve constrained composition.
5. Restructure the main paper around those results and keep the correction ledger supplementary.

This review did not rerun cluster experiments, recruit readers, or obtain new model judgments. The draft has not been newly typeset, so no current page-count claim is made.

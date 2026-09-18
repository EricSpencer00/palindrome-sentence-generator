# Quarantined historical revision response

This response concerns a retired structural-search manuscript. It is not part
of the target readable-palindrome paper and must not support a release.

The manuscript retains the existing definitions, examples, and explanatory
order. The 11 September revision adds the requested controlled search study and
formalizes the solution-preservation argument. It remains a structural-search
paper, not a readability claim.

| Criticism | Change or remaining limit |
|---|---|
| 1. Prior work preempts the novelty claim | The title and introduction identify the contribution as a solution-preserving gate for this two-ended state representation. Related work credits Papadopoulos et al.'s existing POS-pattern constraints. A direct common-baseline experiment remains absent. |
| 2. 4.12 is a single-run observation | The main result is now 50 paired fixed-opening trials with a common maximum popped-state budget, deterministic arm-independent ordering, fresh processes, CPU time, peak RSS, generated/pushed/popped counts, retained outputs, stopping reasons, and paired bootstrap intervals. The old 4.12 observation is explicitly exploratory. |
| 3. “Syntactic” is too strong | Title and section heading use Brown POS shapes. The text states exactly what is admitted, and retains the false-positive example. Dependency structure, agreement, arguments, meaning, disambiguation, and held-out validation are not supplied. |
| 4. Additional diversity is misstated | Full incremental arm: 30 families. Incremental-only: 69,745 pairs in 29 families. Shared: 16,766 pairs. The independent checker now computes both exclusive subsets. |
| 5. Two disconnected studies | Composition and dictionary length are in appendices. They are absent from the title and abstract, are explicitly labeled unrelated to the controlled comparison, and do not support the structural comparison. The inventory table remains individual observations, not a complete ablation. |
| 6. Language usefulness is unmeasured | Scope is structural candidate supply. No grammaticality, readability, preference, coherence, or usefulness result is claimed. A new language evaluation remains absent. |
| 7. Brown count mislabeled | Manuscript and checker distinguish 49,815 word types from 53,548 word-tag associations. |
| 8. Closure label | “Eligible closures (20--44 letters)” and the checks that follow are explicit. |
| 9. Pruning counter | “POS-shape gate rejections” and the excluded restrictions are explicit. |
| 10. Saved report command | Both commands include an explicit output file. The release validator runs these commands after extraction. |
| 11. Missing Universal POS citation | Added Petrov, Das, and McDonald 2012 using the ACL Anthology record. |
| 12. Rerun and source provenance | Added portable current-source rerun commands, frozen inputs, current source file hashes, dirty-tree status, runtime records, and per-process outputs for new structural runs. Historical source/environment gaps remain explicitly unrecoverable. |
| 13. Evidence package incomplete | The evidence archive includes the current runtime source and frozen inputs. Validation extracts it into a clean directory and runs both audits and bounded execution checks. The paper itself contains its central counts, definitions, limitations, and false-positive example. |
| 14. State the central theorem | Section 3 now gives a proposition and proof that the partial gate preserves the terminal solution set under exhaustive exploration. The controlled audit also verifies that every terminal pair reached within the budget appears in its matched incremental arm. |
| 15. Measure search efficiency directly | The controlled table reports generated, pushed, and popped states; eligible closures; accepted outputs; process CPU time; accepted rates; peak RSS; and frontier size. Incremental filtering yields 5.47x accepted pairs per popped state and 4.36x per CPU second, but 0.95x per generated state, so the paper does not claim improvement under every denominator. A secondary exact paired sign test gives p=2.98e-8 for 26 wins, 24 ties, and no losses. |
| 16. Explain the generated-state result | The introduction and experiment section now elevate this from a caveat to a search-behavior finding: pruning redirects a fixed-budget depth-first traversal into deeper, more highly branching regions. The discussion distinguishes conserved resources, while a named finite-label viability proposition formally defines compatible suffix and prefix projections and reserves the empirical claim for POS patterns. |
| 17. Contextualize constrained-search ancestry | Related Work now connects the result to grid beam search, dynamic beam allocation, and lookahead-constrained decoding, while distinguishing those left-to-right approximate policies from this exact two-ended enumerator. |

The private Harvard Dataverse deposit recorded in `DEPOSIT.md` predates the
controlled experiment. The new local release must replace that unpublished
draft before its preview URL represents this revision.

# Reviewer brief

**Revised v1, October 2:** *Fragment Penalties Shift Repetition in Exact Palindrome Search*

The current paper reports 360 new paired AWS search jobs. It tests whether penalizing six frequent trigrams from an earlier collection improves output diversity without reducing exact coverage or increasing other failure modes. Penalties 0, 4, and 16 share 20 fresh seeds, three length bands, two native algorithms, vocabulary, admission gates, and resource ceilings. Penalty order is randomized within each seed–band–method block. No job reaches the time limit.

Each arm produces 90 exact outputs from 120 tasks. Distinct surfaces increase from 69 to 78, mean pairwise trigram distance increases, and the six targets disappear. Replacement trigrams still occur in 46/78 distinct penalized outputs. Strict structural coverage drops from 59/120 to 28/120; this is a separate post-selection mechanical screen, not a grammar score. Two figures show the intervention and recurrence distributions, with paired seed-level uncertainty.

The blinded Qwen3 audit assesses all 148 distinct outputs using the unchanged 0–3 grammar/meaning rubric and exact historical prose/shuffle controls. It is model evidence, not human evaluation. See the [experiment report](../experiments/FRAGMENT-PENALTY-RESULTS-20261002.md) for completed ratings and consistency checks.

The contribution is a measured tradeoff within these implementations. The experiment does not establish superiority over published generators, generalize across vocabularies, or establish human readability. The six targets were selected after inspecting historical data and frozen before the new seed block. Both penalty strengths are reported without selecting a winner.

The old 52-item reader packet is retained unchanged as a historical pilot, separate from these 148 new outputs. Historical provenance audits and the old simulated ARR review are not reviews of the new intervention.

# Paired fragment-penalty audit — October 8, 2026

## Scope

This audit checked the saved October 2 search records, frozen manifest, analysis code, and results write-up. It did not rerun search or judge inference. The experiment supports a within-generator comparison of penalty 0, 4, and 16 under one frozen vocabulary and two search implementations. Penalty 0 is the native scorer wrapped with a zero-strength intervention. These data do not establish an improvement over another generator, a readable-output benchmark, or human-rated language quality.

## Checks and findings

- The 360 manifest jobs and 360 saved rows match by ID and every recorded job field. Each of the 120 seed–band–method blocks contains all three penalties. All 20 planned seeds (7100–7119), all three bands, both methods, and all three strengths are present.
- The frozen manifest source hashes all match the current files. The manifest was frozen at 21:26 UTC on October 2; the recorded search began at 21:31 UTC. The freeze commit `2535cfe9` is dated 21:30 UTC, before search execution. Search records identify Linux, Python 3.11.16, and wordfreq 3.1.1.
- All rows have one of the two expected terminal outcomes: `selected_basic_valid_output` or `no_basic_valid_output`. There are 270 selected outputs and 90 no-output records; no execution errors or deadline exits appear. Every seed–band–method cell has 15/20 successes in each arm. Success and failure task IDs are identical across all three arms.
- Independent recounts reproduce the reported 90/120 exact outputs in every arm; distinct text counts are 69, 78, and 78; strict counts are 59, 28, and 28. Pairwise trigram Jaccard distance recomputed directly from the saved distinct texts is 0.889143, 0.939036, and 0.938402. All 90/90 baseline successes have at least one target fragment; penalized arms have 0/90. The target-free penalized result is balanced by new repeated motifs: each penalized arm's leading trigram occurs in 46/78 distinct texts.
- The paired-seed percentile intervals in the saved analysis are consistent with the implementation: all methods and bands for a seed are resampled together, and conditional diversity is recomputed after collapsing duplicate surfaces. The zero exact-yield difference is a property of these paired records and is not evidence of population equivalence. Intervals quantify seed variation for this fixed setup, not human quality or generalization to other vocabularies.
- The separate blinded Qwen3 records report 0/148 distinct candidates meeting the joint rubric threshold. The control gate passed and hidden repeats agreed 12/12. This is model-rubric evidence only; the experiment contains no human evaluation.

## Analysis hardening

The analyzer previously rejected explicit `execution_error` rows but would have accepted an unexpected terminal status and treated its missing text as a search failure. It also relied on hard-coded seed and strength constants without explicitly checking them against the frozen manifest. It now rejects unknown statuses and inconsistent text/status pairs; verifies the frozen seeds, strengths, and paired-block allocation; and cross-checks each saved row's letter and word counts, exact structural-check keys, basic-to-strict check values, and strict-admission boolean. Failure rows must retain zero counts, false basic checks, and empty strict checks. These are consistency checks over saved records, not a rerun of the admission gate. Reanalysis passes and leaves the reported estimates unchanged. A focused tamper test confirms altered counts, strict outcomes, or check maps are rejected.

## Reproduction

From the repository root, run `python3 -c 'from experiments.analyze_fragment_penalty_20261002 import analyze; analyze(replicates=20000)'` to reproduce the analysis object from the saved records without regenerating figures. The full analysis script also writes the two paper figures. No search compute is needed for either route.

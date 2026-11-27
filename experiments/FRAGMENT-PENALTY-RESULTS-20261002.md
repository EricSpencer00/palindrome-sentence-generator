# Paired fragment-penalty experiment: October 2, 2026

## Finding

Penalizing six fragments selected from the historical collection changed the lexical diversity of new exact outputs. It removed every target, increased distinct surfaces, and reduced trigram overlap without changing basic exact yield. It also reduced strict structural acceptance and left new highly prevalent fragments. No output met the separate model rubric's joint language-quality threshold.

This is a within-implementation intervention. It is not a new readable-generator benchmark, a comparison with published generators, or human evaluation.

## Frozen design and AWS execution

The protocol was committed as `2535cfe9` before execution. It uses 20 fresh seeds (7100–7119), three normalized-letter bands (30–49, 50–79, 80–119), two native search implementations, and penalties 0/4/16: 360 jobs, 120 per penalty. The six targets are the most prevalent trigrams across the prior 52 distinct texts, with alphabetical tie-breaking. Targets, strengths, allocation, gates, resource limits, and endpoints were fixed before the new search; there was no outcome-driven tuning.

The score charges each occurrence, including overlaps, using an exact potential difference. Temporary joins broken by growth are refunded, so cumulative charges equal the final fragment count times the penalty. Exhaustive toy tests verify this in both growth directions. Zero penalty preserves the original scoring increment.

Each seed–band–method block runs all three conditions sequentially in seeded random order on the same worker. All conditions share the frozen vocabulary, gates, beam 256, candidate limit 800, per-parent limit 32, 400-step ceiling, and 180-second cooperative time budget. The AWS c7i.16xlarge worker uses 32 processes. All 360 jobs completed without execution errors or deadline exits. Equal resource ceilings do not imply equal expansion counts; elapsed times and scorer calls are retained.

## Results

| Endpoint | Baseline | Penalty 4 | Penalty 16 |
| --- | ---: | ---: | ---: |
| Basic exact output / requested jobs | 90/120 | 90/120 | 90/120 |
| Distinct surfaces / successful jobs | 69/90 | 78/90 | 78/90 |
| Strict structural pass / requested jobs | 59/120 | 28/120 | 28/120 |
| Mean pairwise trigram Jaccard distance | 0.8891 | 0.9390 | 0.9384 |
| Maximum prevalence of any trigram | 57/69 (82.6%) | 46/78 (59.0%) | 46/78 (59.0%) |
| Successful records with a targeted fragment | 78/90 | 0/90 | 0/90 |
| Distinct texts with a targeted fragment | 57/69 | 0/78 | 0/78 |
| Mean elapsed seconds per job | 2.955 | 3.043 | 3.029 |
| Mean scorer calls per job | 56,415 | 52,679 | 52,722 |

Every method–band cell has 15/20 basic successes, and all arms succeed on the same paired tasks. Penalties 4 and 16 select identical surfaces on 82 of the 90 successful tasks; the remaining eight differ. All failures remain in coverage denominators.

The leading penalized trigrams are “i tai cos” and “no i tai”, each present in 46/78 distinct outputs; neither appears in the baseline distinct set. The recurrence curves inspect all trigrams, not only the frozen target list. No claim is made that a changed complete output is just a resegmentation of the same letter tape: all 83 changed baseline-to-penalty pairs also change their normalized tape.

The strict screen rejects structural shortcuts after selection and does not rerun search. Proper self-palindromic multiword spans fail in 58/90 penalty-4 outputs and 61/90 penalty-16 outputs; self-palindromic words fail in 16/90 for each. Checks can overlap. These properties do not establish grammaticality or human readability.

## Paired uncertainty

The 20,000 percentile bootstrap replicates resample whole seeds (PRNG seed 20261002), keeping methods, bands and arms paired. Conditional diversity is recalculated after collapsing identical surfaces in each resample. Text pairs are not treated as independent trials. These are search-seed variability intervals on one vocabulary, not confidence intervals for human judgments. Distinct counts are finite-collection counts without a bootstrap richness interval.

| Difference from baseline | Penalty 4: estimate [95% interval] | Penalty 16: estimate [95% interval] |
| --- | --- | --- |
| Exact yield, percentage points | 0 [0, 0] | 0 [0, 0] |
| Strict yield, percentage points | −25.8 [−37.5, −15.0] | −25.8 [−37.5, −15.0] |
| Maximum trigram prevalence, percentage points | −23.6 [−37.2, −7.9] | −23.6 [−37.2, −7.9] |
| Mean pairwise distance | +0.0499 [0.0256, 0.0873] | +0.0493 [0.0256, 0.0875] |

The degenerate exact-yield difference reflects identical outcomes in these 20 seed clusters; it does not establish population equivalence. The two strengths are both reported without selecting a winner or conducting confirmatory significance tests.

## Blinded Qwen3 audit

Qwen3 32B on AWS Bedrock receives one text and the existing 0–3 grammaticality/meaning rubric, without penalty, method, source, or paired-output labels. The model is `qwen.qwen3-32b-v1:0` in `us-east-2`; temperature 0, top-p 1, maximum 128 output tokens, no-thinking directive, and structured JSON. The complete 184-presentation inventory was frozen after search completion and before the first new judge call. Ratings never select search outputs.

The control gate passes: 11/12 intact Brown spans and 0/12 corresponding shuffles meet the joint threshold (at least 2 on both dimensions). All 148 distinct candidates are rated once. Twelve hidden repeats—six candidates and six prose controls—agree on both ratings in 12/12 cases.

| Grammar, meaning | Baseline distinct texts | Penalty 4 distinct texts | Penalty 16 distinct texts | Union across arms |
| --- | ---: | ---: | ---: | ---: |
| 0, 0 | 68 | 73 | 75 | 142 |
| 1, 0 | 0 | 3 | 1 | 3 |
| 1, 1 | 1 | 2 | 2 | 3 |
| Joint pass (both ≥2) | 0/69 | 0/78 | 0/78 | 0/148 |

Shared texts receive the same rating across arms, so the arm columns do not sum to the union. Near-floor ratings give no threshold-passing candidate and cannot establish equality of latent language quality or human unreadability. No human ratings were collected.

Seven calls completed through local AWS orchestration before that process was interrupted between requests. The unchanged audit resumed from those receipts on a private AWS c7i.xlarge worker; no successful presentation was repeated except the 12 planned hidden repeats. All inference occurred in Bedrock. Both compute instances terminated automatically, their encrypted disks were deleted, and the task-only buckets, roles, and security groups were removed after verifying local result copies.

## Reproduction and provenance

- `runs/fragment-penalty-20261002/manifest.json`: frozen allocation, source hashes, vocabulary, settings, endpoints.
- `runs/fragment-penalty-20261002/results.json`: all 360 requested jobs, texts, gates, time and scorer calls.
- `runs/fragment-penalty-20261002/quality-manifest.json`: rubric, presentation order, text hashes and control reconstruction references.
- `runs/fragment-penalty-20261002/quality-results.json`: all 184 raw model responses, usage and request hashes.
- `runs/fragment-penalty-20261002/analysis.json`: complete metrics, paired intervals, recurrence distributions and input hashes.

Run `python3 experiments/analyze_fragment_penalty_20261002.py` with `paper/figure-requirements.txt` installed to reproduce the statistics and vector figures offline. The public archive projects author metadata and rebinds hashes; experimental outcomes and ratings are unchanged. Brown corpus text and AWS account/infrastructure records are excluded from that archive. The source experiment keeps its original records intact.

Recorded Bedrock token charges total USD 0.00999480. The conservative local cost estimate is below USD 0.60, using launch-to-verified-termination upper bounds plus storage, IPv4 and a small transfer allowance; this is not a final AWS invoice. The approved task cap was USD 6, with all task resources below USD 10/hour. No temporary AWS resources remain.

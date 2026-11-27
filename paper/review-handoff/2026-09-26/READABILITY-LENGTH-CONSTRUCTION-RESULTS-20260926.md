# Readability, length, and construction results

**Date:** 2026-09-26  
**Status:** offline descriptive analysis of frozen results; no model calls or AWS spend.

This report combines the sourced **Diana frame** 38/54-letter construction
case with frozen v4 language diagnostics, grouped by target length and search
construction strategy.

## Diana frame: more structure, but no human readability result

The 54-letter candidate preserves the inherited “aide / rips / nine / memos;
some / men / inspire / Diana” frame and adds noun-phrase material. It is a
bounded live-residual grammar construction; the 38-letter comparator is an
inherited reference whose authorship is unknown. Both are exact letter-level
palindromes. The 54-letter text is 16 normalized letters,
4 scorer tokens, and 1 sentence longer.
Scorer tokenization splits hyphenated compounds into alphabetic tokens. The
exact texts are:

- “An aide rips nine memos; some men inspire Diana.”
- “An aide rips nine memo-hero memos. Some more home men inspire Diana.”

| Text | Letters | Scorer tokens | Sentences | Unique scorer tokens | Candidate Brown gain | Matched Brown prose gain | Candidate minus prose | Construction provenance |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 38-control | 38 | 9 | 1 | 9 | +0.5152 | +0.0251 | +0.4901 | inherited 38-letter reference/control; this audit does not establish authorship |
| 54-np-candidate | 54 | 13 | 2 | 13 | +0.4071 | +1.9235 | -1.5163 | bounded live-residual noun-phrase grammar intersection over an inherited open frame |

Brown gain is mean log probability minus the mean for 32 same-word shuffles,
in nats per word transition. Both candidates beat their own shuffled baseline.
The prose controls match scorer-token count (9 and 13 tokens) and each has one
sentence; the 54-letter construction has two. The 38-letter reference has
greater Brown order gain than its matched prose control. The 54-letter
construction has 1.5163 fewer nats per transition than
its matched prose control. This is one inherited reference and one extension,
so it cannot show that added length or the construction caused a readability
change. These texts were not among the separate 52 candidates in the Qwen3
audit; its 0/3 ratings do not apply to them. Neither has project human ratings.

## V4 diagnostics by length and construction strategy

The 240-task run contains 20 seeds per resource × length × strategy cell.
The original-v3 resource setting selected 0/120
basic-valid outputs; escalated-v4 selected 66/120
(55%). Every escalated-v4
length × strategy cell selected 11/20. These are construction yields under
the study's basic validity checks, not grammaticality or readability ratings.

| Target letters | Strategy | Original-v3 selected valid | Escalated-v4 selected valid | Escalated actual letters min/median/max |
|---|---|---:|---:|---:|
| 30--49 | center-out | 0/20 | 11/20 (55%) | 32/48/48 |
| 30--49 | outside-in adaptation | 0/20 | 11/20 (55%) | 49/49/49 |
| 50--79 | center-out | 0/20 | 11/20 (55%) | 52/76/78 |
| 50--79 | outside-in adaptation | 0/20 | 11/20 (55%) | 51/79/79 |
| 80--119 | center-out | 0/20 | 11/20 (55%) | 80/88/116 |
| 80--119 | outside-in adaptation | 0/20 | 11/20 (55%) | 80/89/113 |

The 66 static-verifier records were joined to raw construction outputs by
seed, length band, resource setting, and strategy. The analyzer checks that
text hashes and letter counts match. For the readability diagnostics below,
each cell contains 11 successful outputs from the 20-seed frozen v4 task set.
“Actual letters” gives minimum/median/maximum normalized letter counts within
the target band. Brown and GPT-2 columns give candidate-minus-own-shuffle
diagnostics; a positive count means the candidate exceeded its own shuffle
mean. DeepSeek columns count candidate/shuffle/tie choices from one pairwise
judgment per output.

Brown is the add-0.1 word-bigram diagnostic trained on 80% of Brown corpus
documents, with candidates held out. It regenerates 32 item-keyed word-order
shuffles. GPT-2 uses the fixed `openai-community/gpt2` revision
`607a30d783dfa663caf39e06633721c8d4cfcd7e` and scores each candidate against
32 archived same-word shuffles. DeepSeek V3.2 made one counterbalanced
candidate-versus-shuffle judgment per output (Bedrock `us-east-2`, temperature
0, top-p 1); a new request may differ because the provider supplies no seed.
These are different automatic diagnostics and are not combined into a single
readability score. The frozen v4 manifest and scorer code identify the exact
protocol and inputs.

| Target letters | Strategy | n | Actual letters min/median/max | Brown gain, mean (positive) | GPT-2 diff, mean (positive) | DeepSeek candidate/shuffle/tie |
|---|---|---:|---:|---:|---:|---:|
| 30--49 | center-out | 11 | 32/48/48 | +0.584 (10/11) | -0.054 (2/11) | 0/0/11 |
| 30--49 | outside-in adaptation | 11 | 49/49/49 | +0.624 (10/11) | +0.168 (10/11) | 0/0/11 |
| 50--79 | center-out | 11 | 52/76/78 | -0.082 (3/11) | +0.040 (6/11) | 0/0/11 |
| 50--79 | outside-in adaptation | 11 | 51/79/79 | +0.186 (7/11) | -0.009 (6/11) | 0/0/11 |
| 80--119 | center-out | 11 | 80/88/116 | -0.129 (2/11) | +0.005 (6/11) | 0/0/11 |
| 80--119 | outside-in adaptation | 11 | 80/89/113 | +0.099 (8/11) | +0.045 (7/11) | 0/0/11 |

Paired within-seed contrasts compare outside-in minus center-out for the 11
seeds in each band where both methods produced an output. They are descriptive
subgroup summaries; no band-wise significance test was prespecified.

| Target letters | Paired seeds | Brown difference, mean (positive pairs) | GPT-2 difference, mean (positive pairs) |
|---|---:|---:|---:|
| 30--49 | 11 | +0.039 (6/11 positive pairs) | +0.222 (11/11 positive pairs) |
| 50--79 | 11 | +0.268 (8/11 positive pairs) | -0.050 (5/11 positive pairs) |
| 80--119 | 11 | +0.227 (9/11 positive pairs) | +0.041 (5/11 positive pairs) |

In the shortest band, all 11 outside-in outputs stop at the 49-letter ceiling.
The strategy comparison is not stable across diagnostics: Brown favors
outside-in on average in the middle and longest bands, GPT-2 does not show the
same pattern, and DeepSeek ties all 66 candidate/shuffle comparisons. Brown
positive counts for center-out fall from 10/11 in the shortest band to 2/11 in
the longest; outside-in counts are 10/11, 7/11, and 8/11. These are selected,
success-conditioned model diagnostics, not reader ratings or causal effects
of length or construction strategy.

## Reproduction and source hashes

From the repository root, regenerate this report without network access or
inference:

```sh
python3 experiments/analyze_readability_length_construction_20260926.py \
  --output /tmp/readability-length-construction-rerun.md
```

The analyzer validates the frozen sample sizes, length bands, same-seed
pairs, and Diana candidate/control IDs before writing the report. Run it from
the repository root or the extracted reproducibility archive. The current
checkout hash is shown beside the hash of the corresponding public archive
member; some archived JSON files are privacy-projected, as documented in the
supplement manifest. These hashes identify the exact inputs used in each
location; reanalysis from the archive reproduces the reported numeric results.

| Source | Current checkout SHA-256 | Public archive member SHA-256 |
|---|---|---|
| `runs/readability-length-stratified-sentence-aware-20260925.json` | `3a203319b945ecc73fe1ef406ac679819e7f9c1fe6bee2097bdd53617d1f5b23` | `2537f177108f128eb7e29349f6b2073c46d88113b44f88d2300d4aa1d8df9920` |
| `runs/directional-search-resource-sensitivity-20260925/manifest.json` | `f97b76d4094bd5f767a11c3f304ae831edc6ac66df30de5690ddd6b9f8d026cc` | `53cffb4c7efd1a569736f0cbec20da0bec41f4e12245b239adcc5ef2d35f86ff` |
| `runs/directional-search-resource-sensitivity-20260925/confirmatory-v4-resource-sensitivity.json` | `b7f5c9082030903b3e3c29f2ad6e5de3d5fe44da57186e5fecc5ffdc68e76070` | `82a9049a4510adbf83d4d10adea0b105485bd59801a2218e1ade3c084da2be5d` |
| `runs/directional-search-resource-sensitivity-20260925/static-verifier-summary-v4-aws.json` | `b9efa930ec8ff42c7b05175cb0fdda454028f47be281efdb560a1ae7713a5ed4` | `78e705c99fddcaaaa050db4ecea5f30c1e15a93740146d66ace80f9a867c0061` |
| `experiments/analyze_readability_length_construction_20260926.py` | `c0904368caf67d0691c469d03f7e02a19cd71b54192c07bd3cdf799d46862f35` | `c0904368caf67d0691c469d03f7e02a19cd71b54192c07bd3cdf799d46862f35` |

The v4 outputs and Brown/GPT-2/DeepSeek diagnostics are from the same frozen
66-output audit used in the manuscript. The Diana pair is a separate
retrospective case series. No score here is a human readability scale.

# Finding Longer Palindromes and Evaluating Their Readability

Read the [anonymous paper](../output/pdf/naacl2027-v1.pdf) or [source](naacl2027-v1.tex). The revised v1 now reports a fresh paired AWS experiment, motivated by recurring fragments in the earlier collection.

Twenty fresh seeds, three length bands, two search implementations, and three penalty strengths produce 360 jobs. The penalty charges for six trigrams chosen before the new runs. It changes the search score; vocabulary, admission rules, and resource ceilings are matched.

| Measurement | Baseline | Penalty 4 | Penalty 16 |
| --- | ---: | ---: | ---: |
| Basic exact outputs / requested jobs | 90/120 | 90/120 | 90/120 |
| Distinct surfaces | 69 | 78 | 78 |
| Strict structural passes / requested jobs | 59/120 | 28/120 | 28/120 |
| Mean pairwise trigram Jaccard distance | 0.889 | 0.939 | 0.938 |
| Highest prevalence of any trigram | 57/69 | 46/78 | 46/78 |
| Selected outputs containing a targeted trigram | 78/90 | 0/90 | 0/90 |

The intervention improves lexical diversity but leaves concentrated replacement fragments and increases structural shortcuts. The strict checks are mechanical, not a language-quality scale. No human ratings or matched published-generator comparison is provided.

The [experiment report](../experiments/FRAGMENT-PENALTY-RESULTS-20261002.md) gives the full results, uncertainty, protocol, and model-audit details. [The readiness plan](NAACL-READINESS-PLAN.md) records venue and claim boundaries. Historical audits and the 52-item reader packet concern the previous collection; that packet is not an evaluation of the new 148 distinct outputs.

Reproduce the current figures and paired seed-bootstrap analysis from saved data, without AWS or model calls:

```sh
python3 experiments/analyze_fragment_penalty_20261002.py
```

Use the versions in [figure-requirements.txt](figure-requirements.txt). The [evidence archive](v1-reproducibility.zip) includes frozen protocols, results, source, prompts, responses, and anonymous hash links. The [professor handoff](review-handoff/NAACL2027-PROFESSOR-HANDOFF-v1.zip) contains the current paper and source. Brown corpus text and account/infrastructure identifiers are excluded from the evidence archive.

Earlier diagnostic figures remain reproducible with `python3 paper/build_diagnostic_figures.py`. They motivated the intervention and are not new experimental outcomes. The old PDF named `methods-of-finding-longer-readable-palindromes-v1.pdf` remains a historical snapshot.

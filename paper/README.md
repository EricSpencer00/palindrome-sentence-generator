# Local word order in exact letter-level palindromes

The current anonymous manuscript is [Local Word Order in Exact Letter-Level
Palindromes: A Reproducible Audit](naacl2027.tex). It reports exactness checks
and a Brown word-bigram diagnostic on eleven selected texts. The metric
measures local word order relative to same-word shuffles; it does not measure
human readability. No human ratings were collected.

[NAACL-READINESS-PLAN.md](NAACL-READINESS-PLAN.md) records the readiness gates.
[READABILITY-SCORE-AUDIT.md](READABILITY-SCORE-AUDIT.md) records the score
audit, its sources, limitations, and exact rerun commands. The historic
[REVIEW-LOG.md](REVIEW-LOG.md) contains earlier review rounds and their
remaining research gaps.

## Reproduce the manuscript results

From the repository root, using Python 3.10 or newer:

```sh
python3 paper/audit_week_results.py
python3 paper/check_seam_invariant.py
python3 paper/replay_clause_search.py
python3 -m pip install -r paper/readability-requirements.txt
python3 -m nltk.downloader brown
python3 experiments/score_length_stratified_readability.py \
  --expected-report runs/readability-length-stratified-sentence-aware-20260925.json \
  --output /tmp/readability-length-stratified-rerun.json
python3 paper/build_readability_table.py
python3 paper/export_submission.py
python3 output/naacl-submission/anonymous-evidence/verify_anonymous_evidence.py
```

The score rerun uses a deterministic document-level split of the NLTK Brown
corpus: 401 documents train the bigram model and 99 disjoint documents provide
the matched and length-stratified controls. The saved report records the
tokenized-corpus fingerprint, split fingerprints, held-out sentence offsets,
span hashes, software version, and all scores. `--expected-report` fails if the
corpus, controls, protocol, or scores differ. The manuscript table is built
from all eleven audited outputs, including the inherited 38-letter case.

The anonymous archives are written under `output/naacl-submission/`. The
evidence archive includes source, selected results, the frozen calibration,
and an exactness verifier. The verifier needs only Python's standard library;
recomputing Brown scores also needs the packages and corpus above. The
historical `paper/releases/naacl2027-2026-09-11/` archive is quarantined and
does not match this manuscript.

To build a review PDF from the repository root:

```sh
mkdir -p output/pdf/naacl-ready-review
tectonic --keep-logs -o output/pdf/naacl-ready-review paper/naacl2027.tex
```

[The NAACL 2027 call](https://2027.naacl.org/calls/main_conference_papers/)
sets a four-page short-paper limit. The generated review PDF and anonymous bundles are build artifacts; the LaTeX source,
scoring code, audit reports, and cited data provenance define the reproducible
record.

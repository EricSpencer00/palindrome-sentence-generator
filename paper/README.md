# Growing letter-level palindromes

The goal remains long English prose with exact letter-level symmetry. The
paper, [Creating Longer Letter-Level Palindromes Through Paired
Edits](naacl2027.tex), describes the editing setup and reports its saved
results: a 568-to-752-letter lineage and a separate 672-letter search. The
checks establish exactness, not reader-perceived fluency or coherence.

The complete 752-letter rendering stays in the anonymous evidence archive,
not in the main paper. The lineage is selected, not a census or length record.
[REVIEW-LOG.md](REVIEW-LOG.md) records internal reviews and evidence checks.

## Reproduce the evidence

From the repository root, using Python 3.10 or newer:

```sh
python3 paper/audit_week_results.py
python3 paper/check_seam_invariant.py
python3 paper/replay_clause_search.py
python3 paper/export_submission.py
python3 output/naacl-submission/anonymous-evidence/verify_anonymous_evidence.py
tectonic --keep-logs -o output/pdf paper/naacl2027.tex
```

The audit pins original source-file digests and a repository snapshot. It
checks eleven complete renderings using normalized reversal and a separate
raw-text scan. The lineage table is generated from those recomputed records.
The bounded algebra test covers 54,145 equal-length cases plus 91 context
checks. The independent clause replay reproduces the saved text, digest, and
counters: 9,273 frontier examinations, 35 rejected records, and one accepted
chain. No reader or comparative language-quality result is available.

The exporter creates two allowlisted archives under `output/naacl-submission/`:
expanded Overleaf sources and anonymous evidence. The latter runs without
Git history, network access, credentials, or model calls. It excludes raw run
metadata and private host information. Generated PDFs and these archives are
not part of this source commit.

There are no completed blinded reader results or comparative readability
results. The prospective reader protocol is labeled as a plan. Neither this
draft nor its internal AI reviews establish that the research goal is done.

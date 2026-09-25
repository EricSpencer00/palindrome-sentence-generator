# Growing letter-level palindromes

The paper, [Growing Letter-Level Palindromes with Paired Edits: A Case
Study](naacl2027.tex), documents a saved 568-to-752-letter lineage and a
separate exact 672-letter search result. It prints the 752-letter endpoint,
labels its prose as rough and not reader-validated, and reports a Brown
word-order comparison only as a diagnostic. The lineage is selected, not a
census or length record; these are construction artifacts, not readability
incumbents. The repository retains 568 letters as its working-length
incumbent, and the inherited 38-letter example remains the only established
reader-facing benchmark.
[REVIEW-LOG.md](REVIEW-LOG.md) records internal reviews and evidence checks.

## Reproduce the evidence

From the repository root, using Python 3.10 or newer:

```sh
python3 paper/audit_week_results.py
python3 paper/check_seam_invariant.py
python3 paper/replay_clause_search.py
python3 paper/build_readability_table.py
python3 experiments/score_length_stratified_readability.py
python3 paper/export_submission.py
python3 output/naacl-submission/anonymous-evidence/verify_anonymous_evidence.py
mkdir -p output/pdf/paper-revision
tectonic --keep-logs -o output/pdf/paper-revision paper/naacl2027.tex
```

The audit pins original source-file digests and a repository snapshot. It
checks eleven complete renderings using normalized reversal and a separate
raw-text scan. The lineage table and representative 752-letter rendering are
generated from those recomputed records.
The bounded algebra test covers 54,145 equal-length cases plus 91 context
checks. The anonymous verifier also reconstructs the separate 568-to-630 seam
edit and checks that each one-sided 599-letter variant first fails at the
recorded normalized offsets. The independent clause replay reproduces the
saved text, digest, and counters: 9,273 frontier examinations, 35 rejected
records, and one accepted chain. The Brown diagnostic uses a fixed
document-level split of the NLTK Brown corpus and requires `nltk`, `wordfreq`,
and the NLTK `brown` resource. The saved comparison reports the same 38,498
exact candidates for online-residual and offline reverse-pair enumeration under
the finite grammar. The quick verifier checks the archived set and counters;
both source implementations and sanitized inputs are included for a full
rerun. Operator counts are not runtime comparisons.
Install with `python3 -m pip install nltk wordfreq` and
`python3 -m nltk.downloader brown`. It compares
each text with deterministic shuffles of the same tokens; its output is a
local-order diagnostic, not a readability rating. The saved calibration JSON
contains the scores and hashes but not the held-out prose passages.

The current review PDF is `output/pdf/paper-revision/naacl2027.pdf`. The
tracked `output/pdf/naacl2027.pdf` is a separate legacy snapshot and may not
match the current source. The command above builds only the current review PDF;
PDFs are excluded from this source update.

The exporter creates two allowlisted archives under `output/naacl-submission/`:
expanded Overleaf sources and anonymous evidence. The latter includes the
scorer source and can recompute recorded Brown scores when the NLTK Brown
corpus is installed. Its command uses `--expected-report` to fail if corpus
fingerprint, held-out spans, or scores differ; the standard-library verifier
checks saved exactness and tape replays without corpus access. The archive also
labels historical date-like run suffixes as identifiers, not run dates. It
excludes raw run metadata and private host information. Generated PDFs and
these archives are not part of this source commit.

The tracked archive under `paper/releases/naacl2027-2026-09-11/` is a historical
snapshot and does not match this working manuscript. For current review, use
only the freshly generated bundles under `output/naacl-submission/`.

The selected evidence archive carries a replayable normalized-character diff
for each 568-to-752 transition. Those diffs replay the saved tapes, not the
original candidate-generation procedures.

There are no completed blinded reader results. The Brown comparison is
descriptive and does not replace reader judgments. Neither this draft nor its
internal AI reviews establish that the research goal is done.

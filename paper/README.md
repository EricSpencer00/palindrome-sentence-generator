# How to Find a Palindrome That Reads

The current manuscript is `naacl2027.tex`. It is a 15-page research version,
including the appendix and references, revised on 7 September 2026. The filename
is retained for the existing project; this is not a conference page-limit claim.

The revision includes the audited 90,937-letter Norvig comparison, the newer
sentence-planning experiments, an eight-vocabulary placement check, and a frozen
seam-order experiment. Some of the old conclusions did not survive the audit:
the first-seam step function, the 1.09 constant, and the 2,900x extrapolation are
removed. Longer output still does not mean readable output.

Build from the repository root:

```sh
.venv-v3/bin/python paper/build_revision_tables.py
tectonic -X compile paper/naacl2027.tex
pdfinfo paper/naacl2027.pdf
```

The source uses ordinary LaTeX packages and BibTeX. On Overleaf, select
`naacl2027.tex` as the main document and use XeLaTeX. Upload the files in
`overleaf-revision-2026-09-07.zip`; the generated tables are included, so Overleaf
does not need to run Python. Preserve any unrelated files in the project.

`build_revision_tables.py` reads saved results. It does not call models or rerun
search. The old `make_figures.py` belongs to the archived short draft and is not
the current build command. It uses historical temporary files and old plot
labels, so do not use it to regenerate this revision.

The prospective packet lives in `runs/revision-2026-09-07/`. It was frozen before
new model evaluations. Exact-answer failures and truncated replies stay visible.
There is no newly recruited human panel or independent human punctuation arm.

`pre-revision-2026-09-07.tex` preserves the previous short manuscript.
`REVISION-2026-09-07.md` maps the reviewer comments to changes and remaining gaps.
All five model attempts are recorded. The final PDF has 15 pages; its page
count, rendering, and text bounds were checked. Updating the remote
Overleaf project additionally requires an authenticated editing session.

Release audit: see `SOURCE-AUDIT.md`. Run `python3 paper/build_release.py` to rebuild
the Overleaf ZIP and separate evidence ZIP with SHA-256 manifests. The evidence
archive preserves repository paths; the Overleaf archive contains only build inputs.

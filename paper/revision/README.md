# Palindrome paper and figure sources

This source bundle accompanies *Methods of Finding Longer, Readable Palindromes*. The manuscript uses anonymous ACL review formatting and contains five pages, including references and appendices, with main content within the four-page short-paper limit.

## Build

From the bundle's `paper` directory:

```sh
mkdir -p ../output
tectonic -C --keep-logs --outdir ../output naacl2027.tex
```

The cached Tectonic build was verified locally. The included ACL style and bibliography files also support a pdfLaTeX/BibTeX build.

## Reproduce figures and examples

Python 3, matplotlib 3.10.9, and numpy 2.4.1 were used. From `paper`:

```sh
python3 revision/build_figures.py
```

The script verifies file and text hashes, all 360 job identities and selected palindrome streams, saved judge-response JSON, printed example streams, historical target selection, and the separate 32-job development probe. It writes four vector PDFs in `fig/` and the recomputed values in `revision/derived-data.json`.

The paired bootstrap uses 20,000 replicates, resampling the 20 seeds while retaining methods, bands, and arms together and recomputing diversity after duplicate collapse. The additional graphs display finite-collection counts. Reproduction uses saved records and requires no search, inference, or corpus downloads.

## Evidence provenance

The main evidence is the frozen October 2 collection: 360 search jobs and model ratings of 148 distinct candidates. The October 8 probe is separate development evidence: 32 jobs, 16 exact outputs, zero strict passes, and an 824-letter maximum, without readability ratings. Brown control text is excluded; presentation hashes and saved responses are retained.

`evidence/SOURCE-HASHES.json` records original repository-relative source paths, original SHA-256 hashes, and hashes of the published copies. The public probe projection omits the execution hostname and obsolete setup-failure notes and rebinds its manifest hash. Candidate text, outcomes, search settings, model responses, and ratings are preserved.

## Examples

“Never odd or even.” is the manuscript's established illustration. “No rider sees red iron.” is an illustrative calibration example with unverified novelty. “Liam sees mail.” and “No evil did live on.” are established examples. All four are used to explain normalization and construction. The two experimental examples are records `s7100-b1-center-p0` and `s7100-b1-center-p4`; each has 48 letters, passes the strict screen, and receives grammar and meaning scores of zero. The script verifies every displayed normalized stream.

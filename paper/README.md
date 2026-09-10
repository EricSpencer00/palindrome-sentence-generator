# Palindrome paper

The current working paper is `eric_evidence_release_draft.md`. It is the
mechanism-first Markdown draft and includes the evidence-release ledger for
every retained, corrected, and missing result.

`naacl2027.tex` is the archival 15-page typeset revision from 7 September 2026.
It remains in the release because it documents the older full research report;
it is not the canonical working draft. `eric_short_working_draft.md` and files
under `paper/archive/` are working or historical material.

## Release layout

Release products live under `paper/releases/<release-id>/`:

- `source.zip` contains the Markdown draft, the archival typeset source,
  bibliography, and generated tables.
- `evidence.zip` preserves repository-relative evidence paths and SHA-256
  manifests.
- `RELEASE-MANIFEST.json` names both manuscripts, required commands, source
  hashes, external inputs, and known limits.

Temporary renderings and local build products belong under `paper/out/` and are
ignored by Git. The generated `revision-*.tex` tables remain beside the archival
TeX source because that document inputs them directly.

Build and validate the default release from the repository root:

```sh
python3 paper/build_release.py --release-id revision-2026-09-10
python3 paper/validate_release.py --release-id revision-2026-09-10 --compile
```

The validator verifies both archive manifests and compiles the extracted source
bundle in a temporary directory. The evidence bundle does not redistribute
external corpora. `SOURCE-AUDIT.md` records their input hashes, versions,
provenance, and missing evidence.

To render the archival typeset revision without putting products beside its
source, run this from `paper/`:

```sh
tectonic --outdir out/local naacl2027.tex
```

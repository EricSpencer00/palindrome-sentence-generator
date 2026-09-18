# Quarantined historical Dataverse deposit

This file records a historical, unpublished deposit whose claimed paper and
evidence no longer meet the project acceptance standard. It is not a current
NAACL paper, must not be refreshed, cited as current evidence, or submitted.
`prepare_dataverse_upload.py` is intentionally disabled. The remaining text is
retained only to identify prior remote artifacts that must stay quarantined.

The manuscript and local release bundles were refreshed on 12 September 2026
after an information-ordering edit. Their audits pass, but the Dataverse login
session expired before the corresponding draft files could be replaced. The
remote checksums recorded below therefore still describe the 11 September
deposit; `paper/out/dataverse-upload/` contains the pending replacement bytes.

- Dataset ID: `14227064`
- Reserved identifier: `doi:10.7910/DVN/UOTHMD`
- [Owner's draft page](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi%3A10.7910%2FDVN%2FUOTHMD&version=DRAFT)
- Release: `paper/releases/naacl2027-2026-09-11/`

The anonymous preview URL is retained only in the ignored local receipt
`paper/out/dataverse-private-receipt.json`. A signed-out browser check confirmed
that the preview opens, shows all twelve retained files, withholds author metadata,
and displays the current title, description, keywords, and Custom Dataset Terms.
Dataverse preview URLs grant access to every file, however, including restricted
files. The existing preview must therefore be disabled before this record is used
for double-anonymous review.

## Current dataset metadata

Title: **Measuring Reversal Cost in English for Exact Palindrome Search: Paper
and Reproducibility Data**

Description: Reproducibility artifacts for the NAACL 2027 submission
*Measuring Reversal Cost in English for Exact Palindrome Search*. The release
contains the double-anonymous manuscript PDF and source; 900 frozen WikiText-2
spans with derived segmentations and model scores across 36 experimental cells;
controlled POS-shape search trials; exact long-form outputs; independent audit
programs and reports; pinned model, corpus, vocabulary, and software provenance;
licenses; manifests; and rerun instructions. The files distinguish structural
validity and search yield from human language quality.

- Subject: Computer and Information Science
- Keywords: palindromes; natural language processing; language modeling;
  constrained generation; reproducibility; reversal cost; exact search
- License/data-use agreement: existing Custom Dataset Terms retained

## Current release files

The following versioned archival files were uploaded and given file descriptions:

- `naacl2027-mirror-cost-paper-2026-09-11.pdf` (file ID `14228442`)
- `naacl2027-mirror-cost-source-2026-09-11.zip` (file ID `14228441`)
- `naacl2027-mirror-cost-evidence-2026-09-11.zip` (file ID `14228439`)
- `naacl2027-mirror-cost-release-manifest-2026-09-11.json` (file ID `14228438`)
- `naacl2027-mirror-cost-upload-manifest-2026-09-11.json` (file ID `14228440`)

The following double-anonymous review files and updated upload manifest were
then added:

- `naacl2027-mirror-cost-anonymous-review-evidence-2026-09-11.zip`
  (file ID `14228453`)
- `naacl2027-mirror-cost-anonymous-review-manifest-2026-09-11.json`
  (file ID `14228455`)
- `naacl2027-mirror-cost-upload-manifest-2026-09-11-v2.json`
  (file ID `14228454`)

The four older Brown POS-shape files remain in the private draft for history.
They were not replaced or deleted. Both the older `evidence.zip` and the new
archival evidence ZIP retain the project's identifying MIT copyright notice.
They are restricted (file IDs `14227524` and `14228439`); the review-safe archive
contains the same scientific payload with that notice and the Git revision
temporarily withheld. Do not share the active anonymous preview: its bearer token
overrides file restrictions. Disable that preview URL first and provide the
identity-scrubbed archive through the conference's review system.

## Verification

Immediately before upload, `paper/validate_release.py --compile --smoke`
passed portable-path checks, clean-extraction audits, both rerun entry points,
and compilation of the extracted source archive. Dataverse displayed all five
full MD5 hashes on the individual file-metadata pages; each matched the local
v1 or v2 upload manifest. The identity-scrubbed archive additionally passed its
identity scan, payload-hash checks, and clean-extraction scientific audits. The owner page and a
separate signed-out preview both showed `DraftUnpublished` after the metadata
change.

Attribution and upstream terms are recorded in [DATA-TERMS.md](DATA-TERMS.md)
and embedded in the evidence archive. The archive is not offered under a
blanket CC0 waiver.

Sources checked on 11 September 2026:

- [Dataverse 6.10.1: adding a dataset](https://guides.dataverse.org/en/6.10.1/user/dataset-management.html#adding-a-new-dataset)
- [Preview URL for an unpublished dataset](https://guides.dataverse.org/en/6.10.1/user/dataset-management.html#preview-url-to-review-unpublished-dataset)

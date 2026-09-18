# Quarantined historical Dataverse metadata

This metadata describes an unpublished legacy deposit that does not meet the
current acceptance standard. Do not upload, refresh, submit, or cite it.

Target draft: `doi:10.7910/DVN/UOTHMD` (dataset ID `14227064`). Keep this
record unpublished; do not submit it for review.

## Title

Measuring Reversal Cost in English for Exact Palindrome Search: Paper and
Reproducibility Data

## Description

Reproducibility artifacts for the NAACL 2027 submission *Measuring Reversal
Cost in English for Exact Palindrome Search*. The release contains the
double-anonymous manuscript PDF and source; 900 frozen WikiText-2 spans with
derived segmentations and model scores across 36 experimental cells;
controlled POS-shape search trials; exact long-form outputs; independent audit
programs and reports; pinned model, corpus, vocabulary, and software
provenance; licenses; manifests; and rerun instructions. The files distinguish
structural validity and search yield from human language quality.

## Classification

- Subject: Computer and Information Science
- Keywords: palindromes; natural language processing; language modeling;
  constrained generation; reproducibility; reversal cost; exact search
- Terms: retain the existing Custom Dataset Terms; the complete attribution
  and upstream terms are in `DATA-TERMS.md` and the evidence archive.

## Upload set

Run `python3 paper/prepare_dataverse_upload.py`. The directory
`paper/out/dataverse-upload/` contains the archival payload, a separate
identity-scrubbed evidence archive for double-anonymous review, and their
manifests. Versioned names intentionally coexist with the earlier draft files.
The two evidence ZIPs that retain the named project copyright notice are
restricted (file IDs `14227524` and `14228439`). Dataverse anonymous-preview
tokens nevertheless grant access to every dataset file, including restricted
ones. Disable the current anonymous preview before review sharing and provide
the identity-scrubbed archive through the conference's review system instead.

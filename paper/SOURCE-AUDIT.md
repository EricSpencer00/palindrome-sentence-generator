# Source and evidence audit — 7 September 2026

This is a research draft, not a certification of readability or a global record.
All ten active bibliography entries were checked against primary publication
records or the author's/distributor's own material. The archived bibliography
retains earlier, uncited entries; they are not evidence for this revision.

| Reference | Primary source | Scope checked |
|---|---|---|
| Norvig 2016 | https://www.norvig.com/palindrome.html | Version 3, 90,439 letters, 21,012 reported words; original page dates to 2002 |
| Papadopoulos et al. 2015 | https://www.ijcai.org/Proceedings/15/Papers/353.pdf | Authors, pages 2489–2495; conjunction of forward/backward corpus graphs |
| Hokamp and Liu 2017 | https://aclanthology.org/P17-1141/ | Metadata, DOI, lexical constraints and grid beam search |
| Post and Vilar 2018 | https://aclanthology.org/N18-1119/ | Metadata, DOI, dynamic beam allocation |
| Lu et al. 2022 | https://aclanthology.org/2022.naacl-main.57/ | All authors, metadata, DOI, lookahead heuristics |
| Krippendorff 2011 | https://www.asc.upenn.edu/krippendorffs-alpha-reliability | Ordinal coincidence formulation; literature update 2013 |
| Shannon 1951 | https://doi.org/10.1002/j.1538-7305.1951.tb01366.x | Publication metadata and distinction from dictionary segmentation costs |
| Francis and Kučera 1979 | https://icame.info/icame_static/manuals/BROWN/INDEX.HTM | Revised Brown Corpus manual; original 1964 |
| Norvig 2009 | https://www.norvig.com/ngrams/ | Bigram data provenance, distinct from code licensing |
| Speer 2022 | https://github.com/rspeer/wordfreq | Project-recommended Zenodo citation v3.0.2; executed package is 3.1.1 |

## Claims and retained evidence

- **90,937 letters:** `artifacts/norvig-v3/` contains the full text, phrase list,
  result and independent audit. `experiments/audit_norvig_result.py` checks the
  reversal, dictionary membership, unique phrases, endpoints and repetition.
  This exceeds this specific baseline by 498 letters; it is not a readability
  result. Word counts use explicitly different tokenization conventions.
- **Seam-order experiment:** `runs/revision-2026-09-07/` preserves the frozen
  protocol, digest, all five model attempts, raw responses and summary. Missing
  scores remain missing. Calibration contains presentation cues and is not a
  pure semantic discrimination test. Both passing models remain at the nest
  floor; the study does not identify a causal seam-count effect.
- **Conservation:** the same directory preserves all sixteen vocabulary/order
  conditions. Signed debt telescopes; the former universal 1.09 claim is gone.
- **Punctuation:** `runs/punct/after_20b.json` and `after_120b.json` retain all
  six variants of 26 texts. Normalized strings and inserted-punctuation word
  sequences were checked. The +0.77 is a within-model descriptive mean change,
  not a blinded human effect.
- **Scaling:** `runs/polaris/scale_20260823/summaries.jsonl` supports the five
  comparable vocabulary points and slope -1.031. Counts are summed worker-local
  distinct outputs per worker-second, not globally deduplicated output.
- **Sentence planning:** saved aggregates preserve candidate lists and counts;
  raw per-rank traces are not part of this release. Exact finite-template
  results are in `experiments/sentence_intersection-results.json`.
- **Historical seam forced choices:** surviving reports contain totals, but
  their temporary raw per-item files are absent. They are labeled archival.
  Historical punctuation forced-choice counts and an unreproducible numerical
  Poisson interval were removed from the main argument.

No new human panel, independent human punctuation, high-budget length sweep,
controlled beam-width readability curve, or best-of-N curve was conducted.
These gaps cannot be fixed by editing prose. The model results are exploratory.

## Reproduction inputs

The evidence bundle includes SHA-256 hashes of its files. External corpora are
not redistributed: obtain Norvig's dictionary/reference from the links in the
search code, bigrams from the above corpus page, Brown through NLTK, and wordfreq
3.1.1. Preserve upstream attribution and data terms. Installed experiment
versions: Python environment with wordfreq 3.1.1, nltk 3.10.3, numpy 2.4.6.
Table generation and invariant verification use the standard library.

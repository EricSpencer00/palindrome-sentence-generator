# Attribution and terms

The archive combines project code, derived data, and third-party inputs. It is
not offered under a blanket CC0 waiver. Preserve the notices below when using
or redistributing the corresponding material.

- Project source: MIT, with the copyright notice in `licenses/PROJECT-LICENSE.txt`.
- WikiText-2 samples in `data/mirror-cost/results.json`: derived from the
  Salesforce WikiText dataset, which identifies its licenses as Creative
  Commons Attribution-ShareAlike 3.0 and the GNU Free Documentation License.
  The archive includes only the 900 sampled spans and their normalized forms,
  not the source parquet. Dataset citation, snapshot, and source-file hash are
  retained in the result and `licenses/WIKITEXT-NOTICE.md`.
  https://huggingface.co/datasets/Salesforce/wikitext
- Brown-derived `inputs/brown.json.gz`: W. N. Francis and H. Kucera, Brown
  University. The NLTK Brown README permits redistribution; its package record
  specifies noncommercial use. Both are retained under `licenses/`. The derived
  payload stores observed word tags, sentence shapes, and trigram counts.
  Source: https://github.com/nltk/nltk_data/tree/gh-pages/packages/corpora
- Universal POS mapping: Slav Petrov, Dipanjan Das, and Ryan McDonald (2012),
  *A Universal Part-of-Speech Tagset*, LREC, 2089--2096.
  https://aclanthology.org/L12-1115/
  The upstream mapping README is retained under `licenses/`.
- Vocabulary: derived from wordfreq, Robyn Speer, with upstream data and
  attribution notices in `licenses/WORDFREQ-NOTICE.md`. The data license is
  Creative Commons Attribution-ShareAlike 4.0. The frozen vocabulary is a
  filtered English word list; the payload-building environment was not frozen.
  https://github.com/rspeer/wordfreq
- Language-model weights are not bundled. The experiment records exact GPT-2
  and SmolLM2-135M revisions. SmolLM2-135M is distributed under Apache 2.0;
  its model card and paper are cited in the manuscript.
  https://huggingface.co/HuggingFaceTB/SmolLM2-135M
- `inputs/norvig/pal3.py`: Peter Norvig, from pytudes. MIT license retained in
  `licenses/NORVIG-LICENSE.txt`.
  https://github.com/norvig/pytudes/blob/main/py/pal3.py
- `inputs/norvig/npdict.txt`: Peter Norvig's phrase list, derived from Grady
  Ward's Moby Word Lists. The source collection is public domain in the USA.
  https://www.norvig.com/npdict.txt
  https://www.gutenberg.org/ebooks/3201
- `inputs/norvig/pal21txt.html`: Peter Norvig's published version-3 reference,
  retained as a research comparison with its attribution. No new license is
  asserted for that reference. https://www.norvig.com/pal21txt.html

The saved structural outputs and dictionary artifact are supplied for research
verification. Their inclusion does not change the terms of their source inputs.
Copyright and attribution notices remain in the files. A repository's anonymous
preview option hides selected metadata; it does not remove these file notices.

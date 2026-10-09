# Methods of Finding Longer, Readable Palindromes

Read the [revised paper](../output/pdf/paper-revision/palindrome-paper-revised-20261009.pdf), edit the [LaTeX manuscript](naacl2027.tex), or download the [editable source bundle](../output/pdf/paper-revision/palindrome-paper-editable-sources-20261009.zip).

The study compares fragment penalties 0, 4, and 16 across 360 matched search jobs. Each arm produces 90/120 exact outputs. Distinct surfaces increase from 69 to 78, while strict mechanical passes decrease from 59/120 to 28/120. The saved model audit has no joint grammar-and-meaning passes among 148 distinct candidates.

The five-page manuscript contains four vector figures and six verified examples. The examples explain ASCII-letter normalization, paired construction, and the distinction between exactness, structural screening, and readability.

Regenerate the figures and example checks from the saved records:

```sh
python3 paper/revision/build_figures.py
```

See [reproduction instructions](revision/README.md) for dependencies, compilation, and evidence provenance. The two manuscript source names are synchronized. Historical versions remain in Git history.

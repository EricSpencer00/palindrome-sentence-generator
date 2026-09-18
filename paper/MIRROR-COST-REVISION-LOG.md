# Ten critique-and-refinement passes: mirror-cost rewrite

The rewritten manuscript is [`naacl2027.tex`](naacl2027.tex). This log records
the ten skeptical passes used to refine it rather than presenting the result as
an unexamined reframing.

1. **Is this NLP, or a string puzzle?** The initial framing risked leading
   with palindromes as a curiosity. The revision leads with a double
   lexicalization problem: one letter stream must support two English readings.
2. **Is the measurement confounded by the original writer's spaces?** Yes, if
   natural text is compared with a reverse segmentation. Both directions are
   now re-segmented with the same vocabulary and objective, and the introduction
   explains the counterfactual explicitly.
3. **Does a language-model gap establish a linguistic theorem?** No. The
   formula is now described as a causal-LM diagnostic; all information-theory,
   candidate-count, and human-quality claims were removed.
4. **Can a single segmentation objective manufacture the result?** It could
   affect its magnitude. The methods and results now retain unigram, fewest,
   and greedy segmenters and report their separate ranges.
5. **Can related GPT-2 checkpoints establish model robustness?** Not across
   architectures. The text calls them a scale check only and puts a distinct
   model family in the required future replication.
6. **Does dictionary coverage prove readability?** No. The definition excludes
   short-unit tiling, reports coverage as a companion diagnostic, and explicitly
   says full coverage can still be nonsense.
7. **Does the decoder's POS gate make the output grammatical?** No. The
   rewritten decoder section calls it a finite POS-shape language, gives only
   the soundness argument it supports, and does not use it as a grammar claim.
8. **Does the 5.47x result show a universal pruning benefit?** No. It is now
   a bounded, 50-opening candidate-supply case study, with the changed
   generated-state denominator and bootstrap interval retained.
9. **Do the live website and Norvig extension establish NLP significance?**
   No. They now appear only as a system demonstration and an exact-decoding
   stress test. The Norvig comparison is explicitly not a semantic benchmark
   or matched-runtime experiment.
10. **Could a reader mistake this for a prose-quality paper?** The abstract,
    conclusion, limitations, and reproduction appendix now maintain the
    boundary: no human study has established grammaticality, coherence, or
    meaning. The next required work is a held-out, cross-architecture
    mirror-cost replication, followed by blinded human evaluation only if the
    claim shifts to generated quality.

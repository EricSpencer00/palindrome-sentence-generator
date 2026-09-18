# Full-paper critique and revision ledger

Each pass re-read the complete snapshot: title and abstract, introduction,
metric and controls, data and results, decoder experiment, related work,
conclusion, limitations, references, and reproduction appendix. Tables and
reported values were held fixed unless a claim exceeded their evidence. The
final snapshot is `v10.tex`; `../naacl2027.tex` is the polished final version
of that argumentative state, restoring its full abstract and limitations.

## 1. `v01.tex` -> `v02.tex`

The whole paper read as a search-system demo: the title, abstract, contribution
list, discussion, long artifact, and website all privileged scale. The methods
and results supplied no linguistic question, while the decoder and POS study
invited unjustified grammar claims. `v02.tex` changes the paper's object from
“find longer palindromes” to double lexicalization: one letter stream must
support two directional readings; the system becomes motivation rather than
the contribution.

## 2. `v02.tex` -> `v03.tex`

The new introduction had a credible NLP object, but the metric, results, and
appendix still gave no empirical test of it. Decoder yield depended on search
choices and could not measure reversal's linguistic burden. `v03.tex` adds a
reverse-reading experiment over ordinary English and makes the decoder a case
study of its consequence.

## 3. `v03.tex` -> `v04.tex`

The draft compared naturally spaced sources with automatically segmented
reversals, confounding direction with word-boundary treatment. Its metric,
data controls, interpretation, and conclusion therefore could not support the
implied causal reading. `v04.tex` re-segments both directions with the same
vocabulary and procedure, defines mirror cost, and limits it to a causal-LM
diagnostic rather than a grammar, human-preference, or feasibility theorem.

## 4. `v04.tex` -> `v05.tex`

The controlled measure still depended on one boundary objective; the results
and table concealed whether that choice created the reported gap. Related GPT-2
checkpoints were also too close to count as independent robustness evidence.
`v05.tex` carries unigram, fewest-segment, and greedy objectives through the
complete design and calls model variation a scale check, not architecture
replication.

## 5. `v05.tex` -> `v06.tex`

The data, vocabulary, normalization, and model family bounded the claim more
than the title and conclusion admitted. Short-unit fallback could also inflate
apparent lexicalization, while coverage could be mistaken for fluency.
`v06.tex` specifies the WikiText-2/lexicon/GPT-2 scope, uses multi-letter word
coverage, and separates coverage, likelihood, and human-quality claims.

## 6. `v06.tex` -> `v07.tex`

The decoder section still risked translating POS-pattern admission into
grammar. That was unsupported by the method, invalidated by tag ambiguity,
and inconsistent with related palindrome work. `v07.tex` makes the gate a
finite POS-shape feasibility predicate, gives only its prefix/suffix soundness
argument, and keeps exactness, candidate supply, and linguistic quality
separate throughout the paper.

## 7. `v07.tex` -> `v08.tex`

The 5.47x result was under-described: its work unit, paired openings,
traversal, interval, and competing generated-state denominator were needed to
interpret the result. The conclusion otherwise implied a universal efficiency
gain. `v08.tex` reports the 50 paired subtrees, fixed 50,000-popped-state
budget, paired-bootstrap interval, and both denominators, defining the claim
as bounded candidate supply.

## 8. `v08.tex` -> `v09.tex`

The public site and Norvig-derived length extension overwhelmed the controlled
experiment rhetorically. Neither a live demo nor an unmatched long artifact is
a semantic benchmark, adoption result, or search-novelty proof. `v09.tex`
demotes them across abstract, related work, conclusion, limitations, and
appendix to a system demonstration and exactness stress test.

## 9. `v09.tex` -> `v10.tex`

The title, abstract, formula, results table, decoder case study, related work,
and limitations needed one aligned claim. The evidence supports a measured
reverse-reading burden under stated conditions, not universal English behavior
or coherent long-form generation. `v10.tex` makes mirror cost the primary
contribution and preserves the missing frozen spans, second corpus, independent
architecture family, and blinded human evaluation as material limitations.

## 10. `v10.tex` -> final source

The full final manuscript was checked for claim alignment. The metric and
results remain pilot-scale; the decoder establishes only exactness and
finite-pattern soundness; the artifact and website remain subordinate; and no
sentence claims grammar, meaning, or readability. The sole final improvement
adds an explicit research question tying the fixed lexicon, segmenter, and
scorer to lexical coverage and causal-LM cost. The appropriate next study is a
frozen cross-corpus, cross-architecture replication; human evaluation belongs
only in work that claims generated-prose quality.

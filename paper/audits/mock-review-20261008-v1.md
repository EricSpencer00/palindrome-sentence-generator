# Internal mock review — public v1, 2026-10-08

This is a simulated reviewer assessment, not a NAACL decision or official
venue scoring rubric. It assesses the evidence currently available.

## Assessment

A defensible 4/5 recommendation is not established. The current study is a
reproducible negative result: a fixed fragment penalty increases surface
variation while reducing mechanical admission and failing to produce an
output that clears the independent model rubric. The new804- and824-letter exact
outputs do not improve the linguistic contribution.

Strengths: explicit within-implementation baseline, matched fresh seeds,
retained failures and denominators, independent exactness, paired uncertainty,
source/output provenance, and clear distinctions among diversity, structural
screens and reader quality.

Main objections:

1. Readability is neither human-validated nor measured on the long frontier.
   Brown/shuffle control separation cannot establish that transfer.
2. The fixed six-fragment intervention is narrow. Its replacement repetition
   and loss of strict passes make it an informative ablation, not a strong
   generator advance.
3. No matched published-generator comparison establishes external benefit;
   increasing a word cap above an impossible limit is not that comparison.

## Evidence that could change the recommendation

A grammar-preserving construction that improves coherent meaning or achieves
longer intact prose, supported by blinded reader/comprehension judgments;
a frozen held-out comparison at equal resources with a feasible baseline;
and explicit lineage dependence and length-quality tradeoffs. Parameters
selected with Jev or model scores must be chosen on development data, with
final validation untouched. More exact letters alone will not resolve these
objections.

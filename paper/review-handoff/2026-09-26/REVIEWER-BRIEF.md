# Reviewer brief

**Draft:** *Resource Sensitivity in Exact Letter-Level Palindrome Search*

## What the paper establishes

On 120 frozen tasks, a larger bundle of search resources returned 66 basic-valid outputs; the original bundle returned none. This is a bundled, adaptive comparison: beam width, candidate limit, branching, and time budget changed together. The result does not identify which setting caused the increase or show that either search implementation is better.

## What the language checks establish

The automatic diagnostics are descriptive. Brown word-bigram gain was positive for 40/66 selected outputs, GPT-2 likelihood for 37/66, and DeepSeek V3.2 tied every candidate/shuffle pair. A separate Qwen2.5-3B local audit assigned floor grammar and meaning scores to all 52 unique candidates, but it passed only 1/12 initial Brown controls. A post-hoc, individual-sentence sensitivity check passed 4/12 controls. Both miss the frozen 8/12 calibration criterion, so the Qwen candidate scores are inconclusive. The paper has no human ratings.

All 52 unique strings pass the exact normalized-letter reversal check, but none is a word-order palindrome. This letter-level definition does not make a string a coherent sentence. The output set contains 52 distinct strings; 36 of the 66 selected outputs contain the same phrase, “one poem association.” The paper therefore makes no readability, causal, population, or general-superiority claim.

## Questions for review

1. Is this bounded resource-sensitivity result a sufficiently substantial contribution for NAACL Findings?
2. Are the adaptive design and bundled-resource attribution clear enough to avoid a causal overclaim?
3. What minimum additional comparison or human evidence would materially change your assessment, rather than just increase a diagnostic score?

## Current internal estimate

The final independent internal review estimated 3/5 soundness, 2/5 excitement, 4/5 reproducibility, and 2.5/5 overall. This is an informal prediction, not an ARR decision. The most likely concern is contribution size; the next is that the experiment cannot identify which resource setting produced the gain. Reproducibility is supported by a checksum-manifested archive, including the local judge's calibration failure and raw outputs.

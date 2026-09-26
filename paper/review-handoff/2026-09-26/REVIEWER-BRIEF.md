# Reviewer brief

**Draft:** *Resource Sensitivity in Exact Letter-Level Palindrome Search*

## What the paper establishes

On 120 frozen tasks, a larger bundle of search resources returned 66 basic-valid outputs; the original bundle returned none. This is a bundled, adaptive comparison: beam width, candidate limit, branching, and time budget changed together. The result does not identify which setting caused the increase or show that either search implementation is better.

## What the language checks establish

The automatic diagnostics are descriptive. Brown word-bigram gain was positive for 40/66 selected outputs, GPT-2 likelihood for 37/66, and DeepSeek V3.2 tied every candidate/shuffle pair. A local Qwen2.5-3B audit assigned floor grammar and meaning scores to all 52 unique candidates but missed its Brown-control gate (1/12; post-hoc individual-sentence check: 4/12). A separate Bedrock Qwen3 32B audit passed its narrow control screen (11/12 Brown spans versus 0/12 shuffles), but all 52 candidates still scored 0/3 on both dimensions; the matched-shuffle joint-pass rate was also zero, so the paired difference was exactly zero. Seven Brown spans contained multiple sentences; 11/12 hidden repeats agreed. These Brown controls were held out from the Brown scorer's fit; Qwen3's Brown pretraining exposure is unknown, and Bedrock exposed no model-weight digest. These model outputs are not human ratings or evidence of improved readability. The paper has no human ratings.

All 52 unique strings pass the exact normalized-letter reversal check, but none is a word-order palindrome. This letter-level definition does not make a string a coherent sentence. The output set contains 52 distinct strings; 36 of the 66 selected outputs contain the same phrase, “one poem association.” The paper therefore makes no readability, causal, population, or general-superiority claim.

## Questions for review

1. Is this bounded resource-sensitivity result a sufficiently substantial contribution for NAACL Findings?
2. Are the adaptive design and bundled-resource attribution clear enough to avoid a causal overclaim?
3. What minimum additional comparison or human evidence would materially change your assessment, rather than just increase a diagnostic score?

## Current internal estimate

The final independent internal review estimated 3/5 soundness, 2/5 excitement, 4/5 reproducibility, and 2.5/5 overall. This is an informal prediction, not an ARR decision. The most likely concern is contribution size; the next is that the experiment cannot identify which resource setting produced the gain. Reproducibility is supported by a checksum-manifested archive, including both Qwen protocols, saved responses, and the hosted result's offline reanalysis tool.

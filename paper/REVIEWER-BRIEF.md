# Reviewer brief

**Revised v1, October 2:** *Evaluating Exact Palindromes Beyond Word-Order Shuffles*

The current revision frames the work as a focused evaluation case study. Its new retrospective cross-tab shows that 21/66 records, representing 15/52 distinct texts, have positive Brown and GPT-2 likelihood contrasts while receiving zero on both Qwen3 dimensions. The counts reproduce with `python3 paper/audit_diagnostic_overlap.py` from the evidence archive. Every repeated copy agrees on both signs before collapsing by surface-text hash. This adds no inference calls or human ratings. The main text ends on page four; the PDF has five pages including limitations, references, and appendices.

## Study and sample

The manuscript describes 66 selected output records collected from 240 search runs, with 52 distinct strings. It examines the construction records alongside automatic language diagnostics. The four resource settings change together, and the search implementations have different closure rules. The sample therefore establishes neither a resource effect nor an advantage of one method. The target length bands define the collection; subgroup summaries are not a length ranking or a causal length analysis. No published generator was reproduced under matched conditions, and no human ratings were collected.

All 52 distinct strings pass the letter-level reversal check; none reverses word order. Thirty-six of the 66 records contain “one poem association.” Counts refer to output records unless explicitly labeled distinct strings.

## Formal controls and measurements

Brown and GPT-2 compare each candidate with 32 permutations of its own words. These are formal word-order perturbation controls: Brown uses an add-0.1 word-bigram mean log probability contrast in nats per word transition; GPT-2 uses a mean causal log probability contrast per predicted token. Brown's null pools and shuffles the item's word tokens, then repartitions them to the original sentence word counts. GPT-2 uses archived fixed-slot word permutations that preserve the word multiset, punctuation, whitespace, and sentence word-slot counts. The scorer orders differ, and the units are not directly comparable.

In the saved 66-record sample, the candidate exceeds its own shuffle mean for 40/66 Brown contrasts and 37/66 GPT-2 contrasts. This says nothing about an absolute quality threshold, human readability, or relative generator performance. The denominator includes repeated strings.

The Qwen3 32B rubric rates grammaticality and coherent meaning separately on 0–3 scales. It returns 0/52 candidate joint passes, alongside 11/12 held-out Brown prose controls and 0/12 corresponding shuffles passing jointly. The frozen screen uses chosen thresholds of at least 8/12 prose passes and at most 3/12 shuffle passes. These thresholds are a diagnostic sensitivity screen, not externally validated calibration. Seven Brown controls contain multiple sentences; Brown pretraining exposure and hosted model weights are unknown. Qwen2.5-3B fails its control screen, making its candidate scores inconclusive. Neither model audit is human evaluation.

DeepSeek ties all 66 candidate–shuffle comparisons. The separate retrospective Diana construction pair is not part of the 52-candidate Qwen audit and has no project human ratings.

## Prior work and claim boundary

Norvig documents residual-character palindrome search; Papadopoulos et al. describe generation from n-gram corpora; Bonlarron and Régin address constrained RADNER sentence generation with constraint programming and language-model curation. These are relevant prior-work references, not measured baselines in this study. The paper claims no readable-generator advance, no external superiority, and no human readability result.

The historical internal mock review of an earlier resource-centered framing recommended **2.5/5 (Borderline Findings)**. This was an internal assessment of that earlier framing, not peer review and not a score for the current revised manuscript.

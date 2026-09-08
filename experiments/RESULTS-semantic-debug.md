# Semantic order-gain debug screen

Date: 2026-09-04  
Polaris job: `7592102.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov`  
Queue: debug, one Polaris node, 32 ranks  
Wall time: 00:01:27  
Exit status: 0  
Artifacts: `runs/polaris/semantic_debug_20260904_173038/`

## Question

Can a bidirectional word-order signal improve the corrected palindrome search
without destroying closure or diversity? The tested term is conditional bigram
log probability minus unigram log probability. This removes the direct reward
for choosing another common word and measures what the local ordering earned.

All arms used the same 30,000-word vocabulary, 128 seeds, beam 48, per-parent
quota 8, candidate limit 200, minimum 80 letters, and structural-debt weight 2.

## Results

| order weight | closed | distinct texts | openings | attested pairs | mean order gain | mean GPT-2/token* | best GPT-2/token* |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 128/128 | 27 | 5 | 75.6% | -1.927 | -5.850 | -5.631 |
| 0.25 | 128/128 | 12 | 2 | 83.9% | -0.892 | -5.680 | -5.481 |
| 0.50 | 128/128 | 4 | 1 | 89.2% | +0.368 | -5.450 | -5.273 |
| 1.00 | 128/128 | 1 | 1 | 97.3% | +1.355 | -4.653 | -4.653 |
| 2.00 | 128/128 | 1 | 1 | 97.4% | +1.253 | -4.701 | -4.701 |

\* GPT-2 was applied locally to each unique raw word sequence with the same
per-predicted-token normalization for every arm.

## Decision

Do not promote the order-gain scorer as the semantic solution. It preserves
closure and raises both its own adjacency metric and GPT-2 score, but those
gains concentrate the search into a repeated function-word template. GPT-2
therefore confirms the same exploit instead of independently validating prose
quality. The useful setting is 0.25 only as a proposal feature: it gains 8.3
points of attested adjacency while retaining 12 distinct texts. It must not be
the final objective.

The next search change should enforce diversity structurally across starts and
completed texts, then treat semantic scores as filters or Pareto coordinates.
A promotion requires improved held-out order score with no reduction in unique
openings and no repeated-template collapse.

Follow-up completed: `RESULTS-diversity-debug.md` implements this gate and
promotes the constrained low-weight arm as the new research baseline.

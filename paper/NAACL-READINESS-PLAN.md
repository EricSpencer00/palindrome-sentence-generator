# NAACL paper readiness plan

**Active manuscript:** [naacl2027.tex](naacl2027.tex), a four-page, anonymous
report of the v4 exact-palindrome search resource experiment. The earlier
eleven-text local-order audit is supporting retrospective evidence, not the
main experiment. This plan does not claim submission or acceptance.

## Claim boundary

The paper reports that a bundled increase in search resources changed basic-
valid output coverage from 0/120 to 66/120 on this frozen task set. It does not
attribute that difference to runtime alone, rank the algorithms causally, or
claim that the outputs are readable. Brown bigram and GPT-2 measures are
automatic language diagnostics. DeepSeek is one provider-unseeded judgment per
output; its 66 ties are not a human study.

## Work completed

| Step | Evidence/check | Status |
| --- | --- | --- |
| Freeze v4 study and outputs | 240 runs, 20 fresh seed clusters, three length bands, two algorithms and two resource bundles; manifest and result hashes checked | Complete |
| Verify search coverage | 0/120 original, 66/120 escalated; paired difference 0.550, seed-cluster interval [0.350, 0.750]; zero deadline exits | Complete |
| Audit output limitations | 54/66 selected outputs passed the stricter mechanical screen; 26 unique strings per arm; the phrase “one poem association” appears in 36/66 | Complete |
| Run static language diagnostics | Brown and pinned GPT-2 on 66 outputs and 32 same-word shuffles each; one DeepSeek V3.2 comparison per output; requests, responses, settings, hashes and cost preserved | Complete |
| Archive evidence | Deterministic zip with frozen search/scorer inputs and outputs, sources, checksum list, environment files and replay instructions | Complete |
| Revise manuscript | Three-page claim-led draft with diagnostic totals, explicit closure rules, short-word allowlist, bundled-resource attribution, adaptive-design disclosure and limits | Complete; independent reviewer QA complete |
| Reproduce and inspect PDF | Rebuild in a separate output directory, verify three-page count, check extracted text and inspect all rendered pages | Complete; all pages inspected after correcting a TeX percent-sign truncation |
| Recompute diagnostics | Recomputed Brown scores from the frozen inputs, joined saved GPT-2 and DeepSeek outputs, and reran the v4 analyzer with candidate-ID and text-hash checks | Complete; 240 tasks and 66 outputs verified |
| Test implementation | Changed search tests and related palindrome tests | 34 passed; GPT-2 download test needs network; full suite collection is blocked by three historical tests importing absent modules |
| Rerun DeepSeek to assess provider stability | Dependency was added in a temporary directory; the retry could not resolve the AWS sign-in endpoint because DNS/network access is unavailable. The archived 66 responses remain auditable; rerun stability is unmeasured. | Open, non-blocking |

## Evidence and reproduction

The v4 manifest, raw task-level results, model inputs and outputs, source code,
and a per-file checksum list are bundled in `paper/v4-reproducibility.zip`.
The Brown and GPT-2 scorers can be rerun from the frozen inputs. The search
replay checker also verifies that the scorer summary's candidate IDs and text
hashes match the rerun outputs. A fresh DeepSeek call is not guaranteed to
reproduce its response because Bedrock supplied no provider seed.

Search and summary reproduction commands are in
[`DIRECTIONAL-SEARCH-RESOURCE-SENSITIVITY-20260925.md`](../experiments/DIRECTIONAL-SEARCH-RESOURCE-SENSITIVITY-20260925.md).
The prior 11-text score audit, split definition and expected-report rerun are
in [`READABILITY-SCORE-AUDIT.md`](READABILITY-SCORE-AUDIT.md). Historical v1--v3
zero-yield results stay in the run archive and are not pooled into v4.

## Readiness assessment

The current draft is a reproducible, exploratory resource-sensitivity study
with explicit failure and language diagnostics. An independent subagent
reviewer estimated Soundness 3/5, Excitement 2/5, Reproducibility 4/5, and
Overall 2.5 (Borderline Findings); a sympathetic review could assign 3. This
is an informal estimate, not an ARR decision. The limits behind that estimate
remain: one vocabulary, an adaptive multi-parameter bundle comparison, no
human language-quality ratings, and unmeasured provider-response stability.
The evidence does not support presenting an ARR 5 as a likely result.

## External requirements checked

- [NAACL 2027 call](https://2027.naacl.org/calls/main_conference_papers/):
  short-paper length and conference process.
- [ARR author guidelines](https://aclrollingreview.org/authors): reviewers
  assess self-contained submissions and their evidence.
- [ARR review form](https://aclrollingreview.org/reviewform): overall 5 means
  “Consider for Award,” up to the top 2.5%; 3 corresponds to Findings.
- [ARR reviewer guidelines](https://aclrollingreview.org/reviewerguidelines):
  soundness, excitement, and overall recommendation are distinct; claims,
  related work, and reproducibility need explicit support.

# NAACL paper readiness plan

**Active manuscript:** [naacl2027.tex](naacl2027.tex), a four-page, anonymous
report of the v4 exact-palindrome search resource experiment. The earlier
eleven-text local-order audit is supporting retrospective evidence, not the
main experiment. This plan does not claim submission or acceptance.

**24-hour handoff clock:** work began 2026-09-25 22:59 CDT. The draft and
evidence packet are due by 2026-09-26 22:59 CDT; the professor then has at
least seven full days, through 2026-10-03 22:59 CDT. No professor message has
been sent. The review page is ready for a one-author pilot, but its ratings
have not been collected and must not be described as study results.

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
| Archive evidence | 63-member deterministic zip; 62 per-file hashes; v4 model/search records plus both Qwen protocols, runners, result JSONs and report; export-only Brown control projection contains no Brown text | Complete; privacy and hash-link tests pass; archive SHA-256 `0a72885649c102d87fc8b462a66baf5be11b9748300f4726f757c1759b72bc06` |
| Local Qwen audit | 244 presentations using the installed Qwen2.5-3B-Instruct model, frozen scoring protocol and duplicate checks | Complete; model calibration failed, so candidate scores are inconclusive |
| Qwen control sensitivity | Separate 12-sentence/12-shuffle check with unchanged model, prompt, rubric, and gate | Complete; 4/12 sentences and 0/12 shuffles passed; same gate failed |
| Resource-setting attribution follow-up | Frozen five-condition, 600-task OAT protocol; 20 new seed clusters; one setting changed per arm; manifest SHA-256 `3f10b1e14a7282273d647819d130f64b5ce51fac744ce6a6183ad268ccc18733` | Not run; the remote transfer of internal search source and vocabulary was rejected by automatic review. Explicit authorization request is recorded as private ticket #473; no search tasks started. |
| User-provided example score | The pasted sentence matches one v4 candidate's normalized letter tape; its question mark is a punctuation variation. The exact variant received Qwen 3B grammar 0/3 and meaning 0/3; the source candidate received the same scores in the frozen audit. | Diagnostic only and excluded from study totals; it is a letter-level palindrome, not a word-order palindrome. Qwen's failed positive-control gate means neither score is calibrated. |
| Revise manuscript | Four-page anonymous draft with diagnostic totals, Qwen calibration failure, exact closure rules, bundle attribution, adaptive-design disclosure and limits | Complete; final independent reviewer QA complete |
| Reproduce and inspect PDF | Rebuilt in an isolated temporary directory; verified four-page count and extracted text; rendered and visually inspected all pages | Complete; bibliography occupies page 4; TeX emitted line-break warnings only |
| Recompute diagnostics | Recomputed Brown scores from the frozen inputs, joined saved GPT-2 and DeepSeek outputs, and reran the v4 analyzer with candidate-ID and text-hash checks | Complete; 240 tasks and 66 outputs verified |
| Test implementation | Changed search tests and related palindrome tests | 34 passed; GPT-2 download test needs network; full suite collection is blocked by three historical tests importing absent modules |
| Obtain a stronger local/AWS judge | The configured Mini has only the audited Qwen2.5-3B and Gemma 2B cached; the NUC has no Ollama runtime and only smaller cached chat models. AWS has no usable credentials in this session. | No stronger judge is available without a model download/install; none was performed or billed. |
| One-author packet | 52 unique outputs with separate 1–5 grammar and understandability items; local autosave, pause/resume and JSON export are implemented | Ready; ratings not collected |
| Custom `espencer.me` review alias | Current private Tailscale `.ts.net` route exists. No separate review hostname/DNS/Cloudflare Access route is configured; preserve the existing `ssh.espencer.me` route. | Unavailable from current host credentials; use the private Tailscale link for now |

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
with explicit failure and language diagnostics. The final independent
subagent review estimated Soundness 3/5, Excitement 2/5, Reproducibility 4/5,
and Overall 2.5/5 (Borderline Findings). This is an informal estimate, not an
ARR decision. The rating stays at 2.5 because the study uses one vocabulary,
changes multiple resource settings together in an adaptive follow-up, and
contains no human language-quality ratings. It is not credible to present ARR
5 as a likely outcome on the current evidence.

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

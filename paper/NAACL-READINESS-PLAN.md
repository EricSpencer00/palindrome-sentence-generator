# NAACL paper readiness plan

**Active manuscript:** [Evaluating Exact Palindrome Generation Beyond Character Validity](../output/pdf/palindrome-evaluation/naacl2027.pdf),
an anonymous evaluation case study using the existing search and diagnostic
records. The main content ends on page 4; the complete PDF is five pages with
limitations, references, and a separate Diana construction appendix.
The September 26 framing revision runs no new experiments or model scoring
calls. This plan does not claim submission or acceptance.

**24-hour handoff clock:** work began 2026-09-25 22:59 CDT. The draft and
evidence packet are due by 2026-09-26 22:59 CDT; the professor then has at
least seven full days, through 2026-10-03 22:59 CDT. No professor message has
been sent. The review page is ready for a one-author pilot, but its ratings
have not been collected and must not be described as study results.

## Claim boundary

The central result is the coexistence of positive shuffle-relative likelihood
and floor-level absolute model ratings on exact palindrome outputs. Brown
gain is positive for 40/66 selected outputs and GPT-2 for 37/66; Qwen3 assigns
all 52 distinct candidates zero on both rubric dimensions while passing a
small prose-control screen. These diagnostics measure different properties
and have no human validation here.

Construction coverage of 0/120 versus 66/120 supplies the evaluated outputs.
The bundled, adaptive comparison cannot identify the responsible resource
setting. Since the original bundle yields no texts, it provides no comparison
of language quality across bundles. The length table is descriptive; the
Diana pair is a separate retrospective construction case. No prior-system
superiority or readable-generation advance is claimed.

## Work completed

| Step | Evidence/check | Status |
| --- | --- | --- |
| Freeze v4 study and outputs | 240 runs, 20 fresh seed clusters, three length bands, two algorithms and two resource bundles; manifest and result hashes checked | Complete |
| Verify search coverage | 0/120 original, 66/120 escalated; paired difference 0.550, seed-cluster interval [0.350, 0.750]; zero deadline exits | Complete |
| Audit output limitations | 54/66 selected outputs passed the stricter mechanical screen; 26 unique strings per arm; the phrase “one poem association” appears in 36/66 | Complete |
| Run static language diagnostics | Brown and pinned GPT-2 on 66 outputs and 32 same-word shuffles each; one DeepSeek V3.2 comparison per output; requests, responses, settings, hashes and cost preserved | Complete |
| Archive evidence | 71-member deterministic zip; 70 per-file hashes; current manuscript, frozen search/model records, protocols, runners, saved responses, and length analysis; export-only Brown control projection contains no Brown text | Complete; privacy and hash-link checks pass; archive SHA-256 `329631609eabe69d85b5a776f92e8cba94fb1c26ec281fd1525c8a06f08090f2` |
| Local Qwen audit | 244 presentations using the installed Qwen2.5-3B-Instruct model, frozen scoring protocol and duplicate checks | Complete; model calibration failed, so candidate scores are inconclusive |
| Qwen control sensitivity | Separate 12-sentence/12-shuffle check with unchanged model, prompt, rubric, and gate | Complete; 4/12 sentences and 0/12 shuffles passed; same gate failed |
| Hosted Qwen3 rubric audit | 244 Bedrock presentations; 11/12 Brown spans and 0/12 shuffles passed the control gate; all 52 candidates scored 0/3 on both model dimensions and candidate-minus-shuffle joint-pass difference was zero | Complete; 7/12 Brown controls are multi-sentence, repeats agree on 11/12 pairs, and this remains model-only diagnostic evidence. Exact run, costs, hashes, and offline reanalysis: [`AWS-QWEN3-READABILITY-AUDIT-RESULTS-20260926.md`](../experiments/AWS-QWEN3-READABILITY-AUDIT-RESULTS-20260926.md) |
| Readability × length/construction analysis | Offline reanalysis of 66 v4 outputs by target length and construction strategy, plus the 38/54-letter Diana-frame pair | Complete; descriptive Brown/GPT-2/DeepSeek proxies only, no human or causal claim. Source hashes and script: [`READABILITY-LENGTH-CONSTRUCTION-RESULTS-20260926.md`](../experiments/READABILITY-LENGTH-CONSTRUCTION-RESULTS-20260926.md) |
| Resource-setting attribution follow-up | Frozen five-condition, 600-task OAT protocol; 20 new seed clusters; one setting changed per arm; manifest SHA-256 `3f10b1e14a7282273d647819d130f64b5ce51fac744ce6a6183ad268ccc18733` | Not run; the remote transfer of internal search source and vocabulary was rejected by automatic review. Existing private ticket #473 requests authorization; the board is currently unreachable, so its current status could not be confirmed. No search tasks started. |
| User-provided example score | The pasted sentence matches one v4 candidate's normalized letter tape; its question mark is a punctuation variation. The exact variant received Qwen 3B grammar 0/3 and meaning 0/3; the source candidate received the same scores in the frozen audit. Raw request/response, prompt, model digest, decoding settings, and exactness check are recorded in `paper/local-llm-user-example-score-20260926.json` (SHA-256 `849dd1424f57b095638cea3801990addc1328d4baf0d9bc6f9c54abde2044706`). | Diagnostic only and excluded from study totals; it is a letter-level palindrome, not a word-order palindrome. Qwen's failed positive-control gate means neither score is calibrated. |
| Revise manuscript | Evaluation case study with existing length results in the main text, explicit distinction between likelihood and rubric outcomes, and separate Diana appendix | Complete; independent subagent checked claims against saved records and two wording overclaims were corrected |
| Reproduce and inspect PDF | Rebuilt from the current source; five pages including references and appendix, with main content ending on page 4 | Current PDF and professor packet synchronized; all pages rendered and visually inspected |
| Recompute diagnostics | Recomputed Brown scores from the frozen inputs, joined saved GPT-2 and DeepSeek outputs, and reran the v4 analyzer with candidate-ID and text-hash checks | Complete; 240 tasks and 66 outputs verified |
| Test implementation | Changed search tests and related palindrome tests | 34 passed; GPT-2 download test needs network; full suite collection is blocked by three historical tests importing absent modules |
| One-author packet | 52 unique outputs with separate 1–5 grammar and understandability items; standalone HTML, local autosave, pause/resume, import/export, and a close-warning confirmation for failed autosave | Ready; ratings not collected. The intended reviewer has already seen both Qwen judges' aggregate scores and one example score, so the ratings will not be blind to prior model results; the packet states this limitation. |
| Private Tailscale review route | Existing Serve route is `https://erics-mac-mini.taild34f34.ts.net/review`; HEAD returned 200 with certificate verification from the coordinator and Mac mini when resolving the tailnet host to `100.117.97.102` | Private route verified; requires tailnet access; standalone review ZIP is the guaranteed fallback |
| Custom `review.espencer.me` alias | DNS lookup failed from both checked hosts and no route is configured. Standard [Tailscale Serve](https://tailscale.com/docs/features/tailscale-serve) restricts names to the device's tailnet domain. | Not available as requested; no DNS or proxy settings were changed |
| Professor handoff bundle | Deterministic handoff builder includes the PDF, review brief, readiness plan, results report, and a separate method-label-masked rating ZIP; it excludes the full archive containing the sealed key | Built and hash-checked; optional live link is private; offline ZIP works without Tailscale |

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

The earlier resource-centered draft received an informal internal estimate
of Soundness 3/5, Excitement 2/5, Reproducibility 4/5, and Overall 2.5/5.
That historical estimate is neither an ARR decision nor a score of the
rewritten manuscript. The current draft has a clearer evaluation contribution
and retains the same evidence limits: one vocabulary, an adaptive resource
comparison, dependent outputs, and no human language-quality ratings.
The framing review checked accuracy and coherence; it did not assign a new
acceptance score. A wording revision alone cannot establish a higher rating.

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

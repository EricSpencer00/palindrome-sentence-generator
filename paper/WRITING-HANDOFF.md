Latest editorial direction — 11 September 2026: the user asked to run the
controlled experiment proposed in the publication assessment. The manuscript
now centers the formal solution-preservation proposition and the 50-opening
fixed-work comparison. Avoid claims about sentence meaning and unnecessary
summaries or editorial phrases such as "the part that matters." The private
Dataverse files predate this experiment and are stale until explicitly replaced.

Current revision update: the supplied critique is addressed in
`REVISION-RESPONSE.md`. The title is now “Sound POS-Constraint Pruning for
Two-Ended Palindrome Search.” This is a controlled structural-search study;
composition and dictionary length are appendices. `DEPOSIT.md` records the
stale private upload. The paper and evidence archive use portable paths and the
release includes the fixed inputs, raw controlled trials, and audit runtime.

# Current paper-writing decisions — 10 September 2026

Current-file update: the user requested a rewrite resembling the Eric draft in
Git history and a LaTeX version for the project's NAACL target (interpreting
"NAACP" in that context). The sole manuscript is now `paper/naacl2027.tex`, with
its compiled PDF at `paper/naacl2027.pdf`. The prose follows commit `ce7b807`
("good draft") and the original `eric_layout.md` outline. It starts with examples,
then explains the construction and constraints. The correction ledger lives in
`paper/EVIDENCE.md`, separate from the paper. The historical critique is in
`paper/CRITIQUE.md`. Older file references below describe previous decisions;
`paper/eric_layout.md` remains unchanged.

- User initially wanted only coaching and no AI-written manuscript words; later explicitly authorized expansion and repeated rewrites in a writing block. Drafting is now authorized, with the user retaining editorial control.
- Preferred voice: direct, concrete, mechanism-first research prose. Definitions/examples/operations precede implications. Avoid aphorisms, rhetorical X-but-Y openings, generic takeaways, progress-history narratives, and inflated claims. User approved the prose of writing block 57319.
- Working title: “Generating Long Palindromes with Syntactic Pruning and Inventory-Aware Search.” It names the paper's two contributions: applying the sentence-pattern test to partial constructions before closure, and adapting the long dictionary search with dynamic inventory estimates. Current scientific framing: palindrome construction under lexical, repetition, and sentence-structure constraints. Readability is a downstream motivation, not an established result.
- Keep model readability evaluations out of this scoped paper. Do not imply their omission establishes improved meaning or that no evaluations exist elsewhere. No human readability result is available. Do not fabricate experiments or tune prose to AI-detector claims. No external AI detector has been run.
- NAACL 2027 short-paper submission: four pages main content; five after acceptance. Exact current draft page count has not been typeset. Preserve detailed reproducibility in a methods appendix where appropriate.
- User provided a detailed 20-point research critique and asked for a rewrite retaining the prose style. Main changes: explicit contributions and method definitions; exact pair identity/POS filter/prefix-suffix gate; honest fixed-stopping-cap comparison; exact Norvig baseline constraint audit; partial existing ablations; relevant literature; reproducibility and missing provenance next to results.
- Current evidence checkout: d24517d6202a7aee4a0a796ac825f6c6bef020e9. This is a current audit checkout, not a proven historical experiment commit.
- Planning evidence: runs/polaris/sentence_plan_20260904_204815/aggregate.json. 20,989 terminal pairs vs 86,511 planned. Ordered token-sequence pair dedup, globally across ranks. Same normalized letters with different segmentation remain different. Frozen payload gives 21,073 Brown-known words and 5,649 retained sentence shapes.
- Stopping caps per arm/rank: 600 seconds, 20M popped states, 10,000 distinct accepted pairs; first cap/exhaustion ends run. 32 ranks, opening index modulo32, rank seeds0–31. Terminal always precedes planned on each rank; changed traversal changes random consumption. Per-rank summaries absent locally. Timings summed19214.13 vs19050.983 rank-seconds. Nodes count popped states; 138,210,789 planned structural rejections occur before pushing and are excluded from that counter.
- Exact artifact check: 90,937 letters, 16,168 unique phrases, max nonexception token count3, no adjacent repeats. Reference:90,439 letters,16,111 unique phrases,9 adjacent-repeat pairs,max nonexception token count14 under ASCII [a-z]+ tokenizer (includes fragments). Objective/reference comparison is historical, not matched runtime or global record.
- Existing partial length comparisons: static45s68,286; unused45s76,979; unused120s88,101; feasible120s88,095; unused/no cap120s88,455; feasible300s90,937. All adapted arms use letter objective; these do not isolate objective change. Do not fabricate a full ablation.
- paper/verify_structural_draft.py rechecks saved pairs, payload counts, reference/output constraints, and hashes without re-running search. The audit is a separate program from the same project; avoid implying an outside auditor.
- Original user notes paper/eric_layout.md must remain unchanged. Prior older generated draft paper/eric_short_working_draft.md retains model outcomes and is not the current scope. New scientific rewrite goes in paper/eric_structural_revision.md and editing block57319.
- No publishing, paid compute, new human recruitment, or remote submission has been requested. Read-only evidence analysis and local writing/reproducibility artifacts are authorized.

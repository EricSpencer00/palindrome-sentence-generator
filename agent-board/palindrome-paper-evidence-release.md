# Palindrome Paper — Evidence & Release

Scope: prepare the structurally scoped palindrome paper for credible review and release. This board separates work that changes a reviewer's conclusion from limitations that are already honestly bounded by the paper.

## Done — P0

### PAPER-00 — Rebuild the paper around one central question

**Labels:** `narrative`, `scope`, `short-paper`

**Why this is real:** the draft currently reads as three adjacent projects: structural pruning, composition, and a Norvig-based length search. It says they are separate, but does not tell the reader why they form one argument. The result is that the qualifications and methods are easier to retain than the contribution.

**Work:**

- Lead the paper with the structural question: can two-ended search reject partial constructions whose word classes can no longer fit a permitted pattern?
- Give composition and the length artifact one explicit role each in that story. If either role cannot be explained in a few sentences, shorten it, move it to the artifact/appendix, or remove it from this paper.
- Rebuild the abstract and section transitions around the method, intuition, and observed result before audit details.

**Done when:** a reader can state the paper's central question, result, and the role of every retained section after reading the abstract and introduction.

### PAPER-01 — Establish one canonical manuscript and output layout

**Labels:** `release`, `reproducibility`, `output-locations`

**Why this is real:** `README.md` names `paper/naacl2027.tex` as the current manuscript, while `paper/WRITING-HANDOFF.md` names `paper/eric_structural_revision.md` as the current rewrite and calls `eric_short_working_draft.md` older. The build also writes PDFs, logs, ZIPs, rendered pages, and machine-readable JSON directly into `paper/`. A reviewer or collaborator cannot tell which source or artifact is authoritative.

**Work:**

- Choose and document one canonical submission source; mark other drafts as archival or working material.
- Route disposable products to `paper/out/<release-id>/` and release bundles to `paper/releases/<release-id>/`.
- Give build and verification commands explicit output arguments; remove hard-coded dated output names.
- Update the top-level README, release README, and artifact map to use the same locations.

**Done when:** a clean checkout can run the documented build without leaving untracked products in source directories, and a new collaborator can identify the submission source and its release artifacts from one README.

### PAPER-02 — Validate the release bundle in a clean extracted directory

**Labels:** `release`, `reproducibility`

**Why this is real:** the current release script produces dated archives, but the archive has not been verified as the single build input after extraction. A release is only useful if its documented main file, generated tables, bibliography, and commands work without hidden local state.

**Work:**

- Extract the Overleaf/source package into an empty directory and compile its declared main document.
- Check that the evidence archive contains every artifact named in the manuscript's artifact map.
- Add a release manifest that names the canonical manuscript, build command, generated tables, evidence archive, and verifier output.

**Done when:** the clean build succeeds and the manifest's paths resolve within the extracted release or are explicitly marked external.

## Done — P1

### PAPER-03 — Decide whether to replicate the planning comparison or narrow it further

**Labels:** `evidence`, `sentence-planning`, `claim-scope`

**Why this is real:** 86,511 versus 20,989 is a valid observed yield under the recorded caps. It is not a general efficiency result: terminal always ran first, pruning changes RNG consumption and traversal, stopping causes are unavailable per rank, and the timing is wall-clock rather than measured CPU work.

**Work:** choose one path.

- **Evidence path:** rerun with counterbalanced arm order, repeated trials, per-rank summaries, pinned code/environment, and a predeclared outcome.
- **Scope path:** keep the result only as an observed bounded-yield comparison; remove or avoid throughput, speedup, and general-performance language.

**Done when:** every numerical claim matches the evidence path chosen, and the table caption says exactly what is counted and what is not comparable.

### PAPER-04 — Recast the Norvig comparison as a scoped length result

**Labels:** `evidence`, `length-search`, `claim-scope`

**Why this is real:** the 90,937-letter artifact is auditable and 498 letters longer under the stated parser, but its search objective, constraints, and inventory bookkeeping differ from Norvig's. Calling it an unqualified “improvement” invites an invalid apples-to-apples reading.

**Work:**

- Use “498 letters longer than the named version-3 reference under the stated evaluation” in title, abstract, results, and conclusion.
- Keep the objective and restriction differences adjacent to the comparison table.
- If inventory policy itself is claimed as a contribution, add a matched original-objective arm; otherwise label the policy rows as descriptive runs rather than an ablation.

**Done when:** no reader can infer a runtime win, global record, optimality claim, or clean causal attribution to the inventory policy.

### PAPER-05 — Make the POS filter's scope impossible to overread

**Labels:** `claim-scope`, `method`, `evaluation`

**Why this is real:** Brown-derived tag sequences supply structural admission, not grammatical agreement, coherence, or readability. “Sentence” wording can overpromise, especially because outputs are lowercased and unpunctuated.

**Work:**

- Prefer “tag-pattern” or “structural” filter where prose might imply grammar.
- State once, prominently, that passing the filter does not establish a sentence or readable language.
- Audit the title, abstract, table captions, composition section, and conclusion for accidental readability inference.

**Done when:** the paper never treats structural admissibility as a human-language outcome.

### PAPER-06 — Give composition an evidence role or remove it from the short-paper argument

**Labels:** `composition`, `scope`, `evaluation`

**Why this is real:** composition is specified, but this draft presents no composition-quality result. Its bank-level deduplication and repetition rules also differ from the pair-search experiment. As written, a reviewer can reasonably ask why it is a contribution rather than implementation detail.

**Work:** choose one path.

- Add a compact, auditable composition result with one clearly non-readability example and a stated endpoint; or
- Move the procedure to an appendix/repository documentation and retain only the identity needed by the paper.
- If retained, define the units of target `T`, whether it is an upper bound or exact target, what a word-length template is, and whether internal-bigram counts exclude component joins.

**Done when:** composition has a declared evidentiary role, and no result table is read as evaluating it when it does not.

### PAPER-07 — Distinguish saved-artifact verification from historical-run reproducibility

**Labels:** `provenance`, `reproducibility`, `evidence`

**Why this is real:** `paper/verify_structural_draft.py` independently checks saved candidates and the length artifact, but it runs against current code and cannot reproduce the historical cluster environment or stopping causes. That supports exact artifact properties, not historical timing or an external audit.

**Work:**

- Publish a provenance record with source revision, dependency versions, machine details, launcher configuration, and known missing traces.
- Keep “separate verifier in the same project” wording; do not call it independent external audit.
- Make new planning runs preserve rank-level summaries and configuration hashes.

**Done when:** artifact verification and historical execution claims are separately labeled in the paper and release manifest.

## Done — P2

### PAPER-08 — Add a worked method example and minimal auditable outputs

**Labels:** `paper`, `reader-aid`, `method`

**Why this is real:** the paper names normalization, mirror pairs, overhang, orientation, closure, midpoint splitting, and output identity before the reader sees a real output from the structural filter. The existing “lived on decaf” example explains reversal, not what the method accepts or rejects.

**Work:**

- Use one saved accepted pair with its permitted tag pattern.
- Show one partial pair rejected by prefix/suffix feasibility, including the growth direction and the unmatched remainder.
- Pick one term—preferably *overhang*—rather than cycling among remainder, overhang, and debt.
- If PAPER-06 retains composition, show one short composed output with its preserved units and label an awkward example honestly if it is representative.

**Done when:** each example is checked by the verifier; the reader can see why the filter prunes without mistaking structural admission for readability.

### PAPER-09 — Complete method and data provenance citations

**Labels:** `citations`, `provenance`, `related-work`

**Why this is real:** the paper should make the inherited Hoey–Norvig mechanism, Brown/NLTK tag mapping, frozen payload, normalization, and data/version boundary easy to audit. This prevents novelty and reproducibility objections without inflating the contribution.

**Work:** cite the algorithmic ancestry and corpus/tagging pipeline precisely; name the frozen payload and its hash; state what is not redistributed and where upstream terms apply.

**Done when:** every external method, data transformation, and retained artifact has a citation or reproducible local reference.

### PAPER-10 — Move audit detail to where it changes interpretation

**Labels:** `writing`, `structure`, `short-paper`

**Why this is real:** the paper currently gives processor model, process binding, seeds, random-number consumption, and clock-check frequency before the main result, while repeated disclaimers interrupt the method. Some qualifications are essential; their current placement makes the paper feel defensive rather than precise.

**Work:**

- Keep beside the table only the stopping limits and the fact that common limits do not imply equal work.
- Move launcher, hardware, binding, seed, clock, and detailed provenance material to Appendix A or the release manifest.
- Consolidate recurring boundary statements. Attribute directly (“we use Norvig's remainder construction”) instead of repeatedly announcing what the paper does not claim.
- Replace opaque phrases such as “elapsed exposure,” “direct audit,” and “inventory bookkeeping” with the quantity or operation they name.

**Done when:** qualifications remain wherever they change a result's interpretation, but the abstract and main method can be read as an argument rather than a sequence of defenses.

## Done — release draft

### PAPER-11 — Package an evidence-led Markdown draft

**Labels:** `draft`, `release`, `claim-scope`

**Outcome:** Added `paper/eric_evidence_release_draft.md`. It keeps the short draft's direct, mechanism-first prose and ends with a release ledger covering the length artifact, structural-yield comparison, composition scope, corrected overhang and scaling claims, seams, model calibration, punctuation, sentence-quality checks, and finite-language diagnostics. `paper/build_release.py` packages the draft in the evidence archive.

**Verified:** `paper/verify_structural_draft.py`, `experiments/verify_revision.py`, release-bundle generation, and archive membership all pass.

## Closed — Not a new task at the current scope

### PAPER-NB-01 — Recruit human raters

**Disposition:** Not required unless the paper claims readability, grammaticality, or coherence. The right current action is to preserve the no-readability claim, not to invent or rush a human study.

### PAPER-NB-02 — Recover missing historical per-rank traces

**Disposition:** A genuine provenance limitation, but not recoverable from the copied run directory. Address it by limiting current claims and preserving traces in any new run.

### PAPER-NB-03 — Full factorial inventory-policy ablation

**Disposition:** Needed only if policy effects are promoted to a main finding. It is not required for a narrow audited length-artifact report.

### PAPER-NB-04 — External third-party audit

**Disposition:** Not needed. A same-project verifier is sufficient if accurately described; calling it external would be misleading.

### PAPER-NB-05 — Prove a world record or a general runtime advantage

**Disposition:** Out of scope. The defensible comparison is to Norvig's named version-3 reference, and the planning result is bounded by its recorded stopping rules.

### PAPER-NB-06 — Redefine output identity because different segmentations count separately

**Disposition:** Not a defect. The ordered token-pair key is already defined; retain it and avoid calling the count a count of unique normalized letter strings.

### PAPER-NB-07 — Add a human or model readability study just to supply examples

**Disposition:** Not required. The needed examples explain the structural method; they should be explicitly labeled as examples, not smuggled in as readability evidence.

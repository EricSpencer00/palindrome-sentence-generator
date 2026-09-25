# Manuscript review record

The rewrite uses a frozen source snapshot,
`2bdd7df301e185e67480f4cc3a0dbec43589b90b`, and eleven selected examples.
Three separate Luna reviewers examined evidence, method correctness, and
skeptical conference-review concerns. These are AI development reviews, not
peer acceptance decisions or human readability data.

## Round 1: inspect, rewrite, verify

The reviewers found that the earlier manuscript centered short constructions
and search speed, understated the long saved outputs, and conflated authored
recipes with searched generation. They also found missing reader data and
insufficient separation from prior residual and word/character graph methods.

The rewrite centers the 568-to-752 sequence, the asymmetric 630-letter edit,
and the searched 672-letter branch. It prints complete examples and states
the source of each lexical proposal. The audit independently recomputes
exactness, digests, length, token counts, and repetition. The bibliography
corrects Norvig's algorithm page to 2002 and discusses the 2015 corpus-graph
method directly. Prepared reader forms are not reported as participants.

The algebra review found that an insertion equivalence needs equal-length
insertions. The proof now states that assumption, includes an unequal-length
counterexample, and has a bounded independent checker. The retained bridge
is explicitly distinguished from a mismatch in the exact parent.

## Round 2: review the rewritten claims and executable evidence

All three reviewers confirmed the principal numbers and lineage. The method
review checked the 630-letter splice and the independent 672-letter replay,
including 9,273 frontier examinations and 35 rejected records (27 chain-gate,
eight residual). The evidence review confirmed the selected exact examples
and warned against interpreting diagnostic repetition measures as fluency.

Remaining fixes applied after this review:

- Distinguish normalized digest/length checks from the independent raw-text
  exactness check.
- Describe the abstract's character obligations without calling the retained
  bridge an existing mismatch.
- State that the insertion theorem does not automatically certify later
  replacement operations; their complete outputs are independently checked.
- Include the independent clause replay in reproduction instructions.
- Create the promised standalone anonymous evidence package and run it.
- Clarify AI-assistance disclosures required at submission.
- Rebuild the PDF after fixing table width, breakable digest strings, and
  quote glyphs; inspect the rendered pages.

## Remaining empirical objections

A final bounded skeptical pass independently ran the anonymous verifier from
the repository and from a temporary directory, checked the archive inventory,
and found no blocking inconsistency in the reviewed claims or bundled evidence.
Its final wording correction distinguishes repository-local checks from the
standard-library-only anonymous verifier. Visual inspection also caught and
fixed a candidate heading stranded at the bottom of an appendix column.

No completed blinded human study or comparative readability experiment exists
in this snapshot. Most long lexical proposals are authored, the searched
grammar is deliberately small, and inherited motifs have incomplete external
provenance. The draft makes no contrary claim. Wording changes cannot resolve
these evidence gaps; the review loop does not establish acceptance or perfection.

## Venue checks

Formatting and disclosure checks used the official
[NAACL main-conference call](https://2027.naacl.org/calls/main_conference_papers/),
[ARR call](https://aclrollingreview.org/cfp), and
[ARR author checklist](https://aclrollingreview.org/authorchecklist).
The anonymous draft includes limitations and assistance disclosures.
Creating this draft does not submit it to a conference.

## Revision and delivery verification

### Round 3: shorten and rebalance evidence

The draft is now four pages including the complete 752-letter endpoint. A
results table reports the paired-edit ablation, 672-letter search yield,
four-edit growth, finite-set operator equivalence, and bounded algebra check.
Two directly relevant constrained-creative-NLG references were added. The
operator count is not presented as a quality baseline or independent sample.

### Round 4: independent re-review

Two skeptical reviewers confirmed the equation proof, scoped table caption,
commit-versus-file-hash wording, and explicit non-record claim. Exact-text
membership in the matched archive was checked directly: the saved 672-letter
rendering occurs once, with the same parent, seam, eight relation triples, and
normalized digest. The 38,498 count is described as finite-set cardinality,
not a sample or success rate. Reviewers agree that this does not establish
comparative language quality. Human ratings and a relevant quality baseline
remain absent; the manuscript makes no readability claim. This review preceded
the final compact revision and refresh of the anonymous artifact bundle.

### Round 5: shorten and strengthen the evidence-to-space ratio

Following the objection that the four-page draft had too much exposition for
its evidence, the full 752-letter rendering was removed from the main text and
retained in the anonymous evidence archive. The main paper is now a three-page
note with one generated table showing all five exact stages from 568 to 752,
including word/type counts, duplicate sentences, and repeated-trigram rates.
These measures are explicitly descriptive diagnostics, not readability
scores. The introduction and related work now distinguish Norvig's
remainder-based algorithm from n-gram corpus enumeration and cite direct
constrained-generation and evaluation work.

The skeptical re-review verified the table and all stated counts against the
saved artifacts. It also caught and corrected the Norvig citation description.
The remaining substantive gaps are unchanged: no blinded reader responses
and no comparative language-quality result. These are named plainly, not
papered over with automatic metrics.

### Round 6: reduce the paper to its evidence

After the objection that the draft was still too long for its results, the
general motivation, repeated scope disclaimers, and redundant related-work
paragraph were removed. The result is a two-page, roughly 1,000-word note with
six directly relevant references. It retains the seam condition and proof,
the complete five-stage 568-to-752 table, the 672-letter search counters,
paired-seam ablation, and bounded-equation checks. The prose explicitly
states that the longest construction is rough and that human ratings and a
language-quality baseline are absent. Skeptical review caught one scope
ambiguity: the lineage has one insertion followed by three mirrored-support
replacements, so the insertion equation must not be read as certifying every
edit. The text now states this distinction and reports full-tape validation
for each descendant.

The revised PDF was rebuilt and both pages were rendered and inspected. All
citations resolve. Non-fatal underfull-box and `lineno.sty` encoding warnings
remain. The expanded Overleaf-source ZIP was extracted to a clean directory
and independently compiled; the table is inlined, so no table input file is
missing. The selected-output audit, seam/algebra checks, 672 replay, bundle
export, independent anonymous-evidence verifier, and secret scan pass. No
Overleaf connector was available, so the bundle was generated and tested
locally but this revision was not uploaded or remotely compiled. The generated
PDF and ZIP bundles are excluded from the source commit.

### Round 7: narrow the scholarly framing

In response to concern that the paper's contribution language outpaced its
evidence, the title and opening now describe a mechanical audit of selected
constructions. The draft no longer presents the local equation as a general
generation method. It reports what was checked: a saved exact lineage, one
bounded search, and finite algebra tests. It explicitly limits these results
to symmetry and replay and states that they say nothing about reader
understanding or language quality. The 38,498-candidate equivalence result
was removed from the main text as a secondary implementation check. Two
skeptical AI re-reviews found no remaining claim-strength or evidence-scope
must-fix; these are internal reviews, not human readability evidence.

### Round 8: use a direct academic frame and natural prose

The title now names the task and operation directly: “Creating Longer
Letter-Level Palindromes Through Paired Edits.” The introduction explains the
character constraint, briefly situates prior work, and states the actual
experiments. Defensive scope language and abstract process labels were
replaced with direct descriptions of the saved lineage, clause search, and
tests. Two skeptical AI re-reviews found the framing proportionate and the
remaining claims consistent with the artifacts. This wording change does not
add reader or language-quality evidence.

### Round 9: add a length-aware prose control

The current draft adds a held-out Brown word-bigram diagnostic to the paired-
edit construction record. Nine selected project outputs from 498 to 752 letters
are compared with intact spans matched by scorer-token count; every matched
prose span has higher local-order gain. A separate set of 108 held-out spans
checks the same shuffle contrast from 16 to about 2,050 tokens. The paper names
this a local-order diagnostic, not a readability score, and separates the
inherited 38-letter reference from project outputs.

Two independent Luna reviewers checked the claims against the saved result and
revised manuscript. They verified the per-output pairs, means, split counts,
and length-control totals. Their revisions prompted a clearer statement that
the score does not assess syntax or meaning, and that the anonymous verifier
cross-checks recorded scores rather than recomputing them without Brown text.
The updated three-page PDF was rendered and inspected; the source and anonymous
evidence bundles were exported, privacy-screened, extracted, compiled, and
verified. These are internal AI reviews, not peer review or human readability
evidence. No blinded human ratings are available.

## Q10 Pass 1

The first fresh skeptic identified three concrete risks: ambiguity about which
prose controls were paired with palindrome outputs, an overly broad impression
from a retrospective and related sample, and incomplete replay instructions for
the Brown diagnostic. The abstract now names the nine paired controls and
labels the comparison descriptive; Table 2 reports each prose-minus-output
gap; the manuscript specifies smoothing, boundaries, OOV handling, and the
item-keyed shuffle. The anonymous archive now carries the scorer source and
recomputation instructions while still omitting Brown text. A rerun from a
clean extraction reproduced the saved candidate scores, control spans, hashes,
and all 108 length-control rows.

Independent re-reviews caught two follow-up issues: the implementation selects
and scores controls in an interleaved loop, so the paper now says span choice
does not use model scores; and the legacy PDF path appeared current in a stale
handoff. The handoff now labels that path historical, while the README names
the current review PDF under `output/pdf/paper-revision/`. A final audit also
found that `/tmp` may resolve to `/private/tmp` on macOS; path metadata now
normalizes resolved paths, and the bundled scorer was rerun successfully from
that location. The current three-page PDF was rebuilt and visually checked.
These are internal Luna AI reviews, not human ratings or peer review. The
human-readability evidence gap remains open.

## Q10 Pass 2

A fresh skeptic found that the saved 568-to-752 chain could not be reconstructed
from the anonymous package, the manuscript implied more generality than the
authored/search hybrid supports, and the Brown scorer treated an entire
multi-sentence item as one sentence. The scorer now inserts sentence boundaries
for candidate text and retains Brown's sentence boundaries for controls; its
shuffle preserves each item's sentence-length vector. The sentence-aware rerun
changes the nine long-output means to $-0.109$ for the selected palindromes and
1.522 for matched prose, with all nine pairs favoring the prose controls. All
108 additional controls have positive gain over their own shuffle means. The
paper now frames the lineage as a hybrid case study and states its retrospective
selection and dependence.

The evidence package now includes a four-edge normalized-character diff that
reconstructs each saved 568-to-752 tape and checks parent/child lengths and
hashes. It is explicitly a tape replay, not a replay of the original generation
procedure. The score report records the NLTK version, training/holdout ID
digests, and a SHA-256 fingerprint of the full tokenized Brown sentence stream;
`--expected-report` makes a score rerun fail if corpus, span, or score values
differ. Date-like suffixes in inherited run filenames are labeled as identifiers,
not run dates. A second reviewer caught the remaining `W/V` table abbreviation
and a stale-PDF wording issue; the table now says `Words/types`, the manuscript
defines the count distinction, and the README labels the tracked PDF as a
separate potentially stale legacy snapshot. The legacy PDF and log were not
changed or staged in this pass.

The focused scorer/table tests passed (10 tests), adjacent exact-audit and
quarantine tests passed (12 tests), the anonymous verifier passed with all 11
exact examples and four lineage replays, and the privacy-screened archive was
rebuilt. The current three-page PDF was rendered and visually checked. A full
scorer rerun from the extracted bundle using the installed Python environment
was terminated by the execution environment (exit 143); the sentence-aware
scores were produced in the repository, and the archived verifier and
`--expected-report` comparison are separately tested. A final independent
Luna review found no actionable remaining defect. Human readability results
and a relevant baseline remain unresolved empirical gaps; internal AI reviews
do not resolve them.

## Q10 Pass 3

A fresh skeptic found that the paper framed a single saved hybrid lineage too
generally, kept the Brown diagnostic too prominent, and hid the actual 752-letter
endpoint in the artifact archive. The title and related-work discussion now
call this a case study, distinguish the authored paired-edit lineage from the
separate bounded search, and state that the seam equation is not a new
palindrome principle. The exact 752-letter endpoint is printed with its
normalized digest and a clear “rough prose, not reader-validated” label.

Independent revision review then identified two missing setup details. The
paper now reports the 672 search's fixed 10-name/3-predicate inventory,
four-clause chains, normalized seam cuts, and 160-key relation-count index,
and labels its evidence as a deterministic replay rather than a general
novelty search. It also states that both matched and length-control Brown
spans use token-count closeness and deterministic SHA tie-breaking, do not
overlap, and are selected without model scores. The chart's contribution is
described narrowly as connected subject-predicate-object clauses filtered
against a reverse character residual that resets at clause boundaries.

The representative-output generation test and adjacent scorer/table/quarantine
tests passed (9 tests). The anonymous verifier checked all 11 exact examples,
the four saved lineage diffs, the 630-letter seam replay, the 672-letter fixed
search replay, and associated diagnostics. The privacy-screened source and
evidence bundles passed; the current three-page PDF compiled, rendered, and
was visually inspected. A fresh final Luna reviewer found no remaining
must-fix. No human readability evidence or comparative language-quality
baseline was added; those empirical gaps remain open. Internal AI reviews are
not reader evidence or peer review.

## Q10 Pass 4

A fresh skeptic found that stale page renders no longer matched the source,
that the selected 752/672-letter artifacts could be mistaken for the project's
working-length incumbent, and that the 9,273 counter needed a precise
definition. The paper and README now distinguish those selected construction
artifacts from the 568-letter working incumbent and inherited 38-letter
reader-facing benchmark. The counter is defined as the sum of live right-clause
frontier sizes before each residual-character filter, not unique search nodes
or candidate chains. The PDF and all three page renders were rebuilt from the
current source. A fresh final reviewer found no remaining must-fix. Readability
is still unverified by people.

## Q10 Pass 5

A fresh skeptic confirmed the separate 568-to-630 ablation's numbers but found
that its one-sided outcomes were not checked by the standalone anonymous
verifier. The exported seam fixture now pins both expected one-sided outcomes,
and the verifier reconstructs each variant from the included 568-letter parent
and independently checks normalized length, exactness, first mismatch offsets,
and mismatching letters. The manuscript now names those outcomes while
restricting the inference to closure at this saved seam; it explicitly makes
no language-quality or general-advantage claim. The final independent review
found no must-fix, confirmed the updated verifier and the current three-page
render, and reiterated that no human readability result exists. The Brown
word-order statistic remains a local-order diagnostic, not a readability
judgment. Internal AI review is not peer review or reader evidence.

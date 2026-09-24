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

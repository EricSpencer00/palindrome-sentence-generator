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

## Delivery verification

The final local manuscript compiles to nine pages including references and
appendices; the main discussion ends on page six. Every page was rendered
and visually inspected. The final build has no overfull boxes, missing glyphs,
or unresolved references. Non-fatal underfull-box warnings remain.

The staged source diff and both allowlisted delivery bundles pass the secret
scanner. A private receipt in an already ignored output directory is excluded.
The original Overleaf sources were backed up locally before replacing only
the manuscript and bibliography. Both replacement uploads were acknowledged
by Overleaf, and recompilation was started. The browser-control connection
closed before its final status could be read; remote compilation is therefore
not claimed as verified. The locally compiled PDF is the verified deliverable.

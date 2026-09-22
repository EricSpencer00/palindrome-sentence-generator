"""Packed ABBA paragraph seam search.

The paragraph is four intact clause slots, A B | B A.  Sentence punctuation is
inside the lexical tape but contributes no matching character.  The NFA pair
intersection advances the two character fronts together; no completed clause
strings are generated before the exact gate.  The middle bar is a topology
label, not a claim that the two B clauses are identical.
"""
from pathlib import Path
import hashlib
import json

from experiments.packed_seam_grammar_20260927 import Grammar, intersect, norm, audit
from llm_palindrome.validator import is_palindrome

ROOT = Path(__file__).resolve().parents[1]
ID = "packed-abba-paragraph-seams-20260928"

# Intact, ordinary clauses.  A/B banks are deliberately distinct so a closure
# cannot be obtained by repeating one self-palindromic unit.
A = (
    "the quiet clerk files a report.", "an old nurse reads the note.",
    "a patient editor sorts the letters.", "the young poet keeps a journal.",
)
B = (
    "the careful aide sends a memo.", "some writers praise the actor.",
    "a senior teacher marks the essay.", "the kind doctor helps a child.",
)


def paragraph_grammar():
    g = Grammar()
    # Four independently realized clause slots. The role labels record the
    # ABBA seam for provenance; they do not force lexical identity.
    g.slot(A, "A:left")
    g.slot(B, "B:left")
    g.slot(B, "B:right")
    g.slot(A, "A:right")
    return g


def run():
    result = intersect(paragraph_grammar(), max_letters=220, cap=120000)
    rows = []
    for row in result["candidates"]:
        text = row["rendered"]
        check = audit(text)
        independent = is_palindrome(text)
        assert check["exact"] and independent
        row.update(audit_two_pointer=check, independent_validator_exact=independent,
                   provenance_path="A:left -> B:left -> B:right -> A:right",
                   seam_topology="ABBA", novel_relative_to_seed=True,
                   human_readability_evidence="not collected",
                   reader_status="unreviewed; no automatic metric certifies prose")
        rows.append(row)
    result["candidates"] = rows
    result.update(
        experiment_id=ID,
        method="live character-level intersection over four intact clause slots",
        topology="A B | B A; two intact clauses on each side of the middle seam",
        source_banks={"A": list(A), "B": list(B)},
        controls=[
            {"rendered": "The quiet clerk files a report. A careful aide sends a memo.",
             "audit": audit("The quiet clerk files a report. A careful aide sends a memo."),
             "kind": "intact ordinary prose, non-palindrome"},
            {"rendered": "The quiet clerk files a report. A careful aide sends a memo. A careful aide sends a memo. The quiet clerk files a report.",
             "audit": audit("The quiet clerk files a report. A careful aide sends a memo. A careful aide sends a memo. The quiet clerk files a report."),
             "kind": "ABBA surface control, not an exact palindrome"},
        ],
        provenance=dict(complete_sentence_enumeration=False, catalogue_replay=False,
                        repeated_self_palindromic_units=False, semordnilap_bank=False,
                        per_candidate_rlaif=False,
                        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
        novelty_preflight=dict(novel_algorithm_claim=False,
            representation_change="paragraph-level four-slot ABBA seam with independent B realizations",
            exact_gate="two-pointer audit plus llm_palindrome.validator.is_palindrome plus forward/reverse SHA"),
        reader_test=dict(status="not collected",
            next="If a novel exact survives, randomize blinded intact and shuffled controls before calling it readable."),
        next_repair=dict(operator="Replace one A/B clause bank with a typed two-clause scene lattice while retaining the same live seam product.",
                         reason="Current residual has no exact closure; widening unrelated lexical banks would duplicate search rather than change topology.",
                         concrete="Add subject/verb/object agreement states and a second event per side, then rerun the same ABBA product."),
    )
    return result


if __name__ == "__main__":
    result = run()
    (ROOT / "runs" / f"{ID}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"states": result["states"], "transitions": result["transitions"],
                      "exact": len(result["candidates"]),
                      "max_letters": max((r["audit"]["letters"] for r in result["candidates"]), default=0)}))

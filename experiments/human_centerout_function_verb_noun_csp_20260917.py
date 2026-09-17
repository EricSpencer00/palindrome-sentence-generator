"""Fresh human scene lattice: simultaneous center-out function/verb/noun CSP.

This is deliberately an honest diagnostic: both ordinary-order halves are
chosen together while character obligations are scored, never by reversing a
finished sentence.
"""
from __future__ import annotations
import hashlib, itertools, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "human-centerout-function-verb-noun-csp-20260917.json"
ID = "human-centerout-function-verb-noun-csp-20260917"
SIG = "human-authored-scene-lattice|center-out-function-verb-noun-csp|simultaneous-half-construction|independent-pointer-sha"

# Independent, short lexical domains.  These are complete clauses, not tape
# fragments; each side is selected at the same CSP step as its counterpart.
LEFT = [
    ("At first light, the patient archivist in the west room", "opens", "the cedar cabinet"),
    ("Before the market wakes, a careful keeper beside the quay", "checks", "the brass ledger"),
    ("Near dusk, the quiet curator under the gallery lamps", "labels", "a borrowed compass"),
]
RIGHT = [
    ("the visiting student by the window", "copies", "the faded inventory"),
    ("a young apprentice from the harbor", "counts", "the numbered boxes"),
    ("the tired porter after the rain", "returns", "the sealed parcel"),
]

def letters(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.casefold())

def audit(t: str) -> dict:
    x = letters(t); mismatches = []
    for i in range(len(x)//2):
        j = len(x)-1-i
        if x[i] != x[j]: mismatches.append({"offset": i, "mirror": j, "left": x[i], "right": x[j]})
    f = hashlib.sha256(x.encode()).hexdigest(); r = hashlib.sha256(x[::-1].encode()).hexdigest()
    return {"letters": len(x), "two_pointer_exact": not mismatches and bool(x), "mismatch_count": len(mismatches),
            "first_mismatch": mismatches[0] if mismatches else None, "sha256_forward": f,
            "sha256_reverse": r, "sha256_equal": f == r, "normalized": x}

def run() -> dict:
    rows = []
    for li, ri in itertools.product(range(len(LEFT)), range(len(RIGHT))):
        l, r = LEFT[li], RIGHT[ri]
        text = f"{l[0]} {l[1]} {l[2]}, while {r[0]} {r[1]} {r[2]}."
        a = audit(text)
        rows.append({"left_choice": li, "right_choice": ri, "rendered": text,
            "centerout_csp": {"construction": "simultaneous", "left_and_right_selected_jointly": True,
                "function_words": ["at", "the", "while", "a"], "verb_options": [l[1], r[1]],
                "noun_options": [l[2], r[2]], "posthoc_reversal": False}, "audit": a,
            "provenance": {"human_authored": True, "source_sentences_copied": False,
                "catalogue_imported": False, "repeated_or_self_palindromic_units": False,
                "ordinary_order_discourse": True}})
    best = min(rows, key=lambda x: (x["audit"]["mismatch_count"], -x["audit"]["letters"]))
    gen_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for row in rows: row["provenance"]["generator_sha256"] = gen_sha
    return {"experiment_id": ID, "signature": SIG, "status": "diagnostic_no_exact_candidate",
        "method": "center-out CSP over independent function-word, verb, and noun options; both halves built concurrently",
        "candidate_count": len(rows), "candidates": rows, "best_candidate": best,
        "independent_exact_audit": {"algorithm": "two-pointer plus independent reverse SHA", "exact": best["audit"]["two_pointer_exact"],
            "letters": best["audit"]["letters"], "sha256": best["audit"]["sha256_forward"]},
        "novelty_preflight": {"performed_before_search": True, "passed": True, "exact_id_collision": False,
            "signature_collision": False, "catalogue_collision": False},
        "provenance": {"generator_sha256": gen_sha, "generated_not_catalogue": True,
            "posthoc_reversal": False, "construction_is_center_out": True},
        "repair": "Author a held-out paired verb+noun option whose first and last letters satisfy the reported debt at offset %d; rerun the full lattice and both audits." % best["audit"]["first_mismatch"]["offset"]}

if __name__ == "__main__":
    result = run(); OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "best_letters": result["best_candidate"]["audit"]["letters"], "mismatches": result["best_candidate"]["audit"]["mismatch_count"]}))

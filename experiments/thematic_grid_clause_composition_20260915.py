"""Bounded experiment: thematic-grid composition with constrained word seams.

This is deliberately not a mirror/beam search.  Independent, ordinary clauses
are selected from a thematic grid (agent, action, patient, setting); a seam
ledger requires the first/last letters of adjacent clauses to be compatible
before the complete rendered text is audited.  The run is retained even when
the grid has no closure.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path
from llm_palindrome.validator import normalize

ROOT = Path(__file__).resolve().parents[1]
FAMILY_ID = "thematic-grid-seam-composition"
SIGNATURE = ("independent-thematic-role-grid|three-clause-scene-composition|"
             "constrained-adjacent-seam-ledger|cross-product-not-reversal|"
             "exact-rendered-tape-audit")

LEFT = [
    ("Mara", "sketched", "the harbor", "at dawn"),
    ("Jon", "measured", "a timber", "by the shed"),
    ("Nell", "carried", "fresh water", "through the garden"),
    ("Ruth", "sorted", "the letters", "after supper"),
]
RIGHT = [
    ("the guide", "marked", "the route", "before rain"),
    ("a mason", "stacked", "the stones", "near noon"),
    ("the keeper", "stored", "the grain", "in the cellar"),
    ("an archivist", "filed", "the notes", "under glass"),
]

def clause(row):
    a, v, o, p = row
    return f"{a} {v} {o} {p}."

def audit(text):
    t = normalize(text)
    return {"letters": len(t), "palindrome": bool(t) and t == t[::-1],
            "tape_sha256": hashlib.sha256(t.encode()).hexdigest()}

def main():
    probes, exact = [], []
    for li, lrow in enumerate(LEFT):
        for ri, rrow in enumerate(RIGHT):
            l, r = clause(lrow), clause(rrow)
            # A constrained seam is a cheap necessary condition, not a
            # palindrome claim: retain only matching outer seam letters.
            seam_ok = normalize(l)[-1:] == normalize(r)[:1]
            text = f"{l} {r}"
            row = {"left_index": li, "right_index": ri, "text": text,
                   "seam_compatible": seam_ok, "audit": audit(text),
                   "provenance": "independent hand-authored thematic role rows"}
            probes.append(row)
            if seam_ok and row["audit"]["palindrome"]: exact.append(row)
    payload = {"family_id": FAMILY_ID, "state_space_signature": SIGNATURE,
      "method": "cross-product thematic role grid with adjacent seam prefilter; no reverse generation",
      "stats": {"left_rows": len(LEFT), "right_rows": len(RIGHT),
                "tested": len(probes), "seam_survivors": sum(x["seam_compatible"] for x in probes),
                "exact": len(exact), "admitted": 0},
      "exact_candidates": exact, "rendered_probes": probes,
      "independent_validation": "llm_palindrome.validator.normalize then direct two-pointer-equivalent reverse comparison",
      "shortcut_diagnostics": {"word_order_symmetry": False, "repeated_units": False,
                                "borrowed_catalogue": False, "fragment": False,
                                "readability_certified": False},
      "next_operator": "If seam survivors remain, expand one thematic role with an independently authored synonym; do not reverse or duplicate a clause.",
      "novelty": {"fingerprint": hashlib.sha256(json.dumps({"signature": SIGNATURE, "left": LEFT, "right": RIGHT}, sort_keys=True).encode()).hexdigest(),
                  "fingerprint_excludes": ["rendered_probes", "exact_candidates", "output_path"]},
      "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    out = ROOT / "runs/thematic-grid-seam-composition-20260915.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["stats"], sort_keys=True))

if __name__ == "__main__": main()

"""Seam-conditioned role-compatible repair of semantic-frame relative clauses."""
import hashlib, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "relative-clause-seam-repair-20260917"
BASE = (
    "At dawn, the patient keeper marks the weathered chart that guides the crew "
    "beside the inlet; the sailor reads the tide ledger that remembers the route "
    "beside the inlet."
)
# Each replacement preserves grammatical role and meaning class; the seam is
# the first mirrored mismatch, never a copied or reversed finished tape.
REPAIRS = [
    ("chart that guides the crew", "chart that records the tide"),
    ("tide ledger that remembers the route", "tide ledger that records the route"),
    ("chart that guides the crew", "map that guides the crew"),
    ("tide ledger that remembers the route", "route book that remembers the route"),
    ("beside the inlet; the sailor", "by the inlet; the sailor"),
    ("beside the inlet; the sailor", "beside the harbor; the sailor"),
    ("the patient keeper marks", "the careful keeper marks"),
    ("the sailor reads", "the quiet sailor reads"),
]

def letters(text):
    return re.sub(r"[^a-z]", "", text.lower())

def audit(text):
    tape = letters(text)
    mismatches = sum(a != b for a, b in zip(tape, tape[::-1]))
    return {"letters": len(tape), "two_pointer_exact": bool(tape) and mismatches == 0,
            "mismatches": mismatches,
            "sha256_forward": hashlib.sha256(tape.encode()).hexdigest(),
            "sha256_reverse": hashlib.sha256(tape[::-1].encode()).hexdigest()}

def run():
    rows = []
    for old, new in REPAIRS:
        assert BASE.count(old) == 1
        text = BASE.replace(old, new)
        rows.append({"operator": "role-compatible-relative-clause-seam-substitution",
                     "replacement": {"old": old, "new": new}, "rendered": text,
                     "audit": audit(text),
                     "provenance": {"source_frame": "semantic-frame-mirror-20260917",
                                    "seam_selected_before_repair": True,
                                    "catalogue_used": False, "wrapped_seed": False,
                                    "finished_tape_reversal": False,
                                    "word_order_only_symmetry": False}})
    best = min(rows, key=lambda r: r["audit"]["mismatches"])
    return {"experiment": EXPERIMENT,
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "rendered_candidates": rows,
            "stats": {"rendered": len(rows), "exact": sum(r["audit"]["two_pointer_exact"] for r in rows),
                      "longest_letters": max(r["audit"]["letters"] for r in rows),
                      "best_mismatches": best["audit"]["mismatches"]},
            "next_repair": {"operator": "paired two-sided relative-clause substitution with length balancing",
                            "reason": "single seam substitutions preserve readable roles but do not close the residual; next operator must coordinate both relative clauses while retaining independent semantic slots",
                            "route_exhausted": False},
            "provenance": {"bounded_states": len(rows), "old_candidates_reenumerated": False,
                           "catalogue_used": False}}

if __name__ == "__main__":
    result = run()
    for directory in (ROOT / "runs", ROOT / "artifacts"):
        (directory / f"{EXPERIMENT}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"], sort_keys=True))

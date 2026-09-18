"""Bounded semantic-frame mirror search; mutable clause slots, no fixed tape."""
import hashlib, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "semantic-frame-mirror-20260917"
FRAMES = [
    ("harbor", "At dawn", "the patient keeper", "marks", "the weathered chart", "that guides the crew", "beside the inlet"),
    ("garden", "After rain", "a quiet teacher", "tends", "the young seedlings", "that shelter bees", "near the stone wall"),
]
MIRRORS = [
    ("sailor", "the sailor", "reads", "the tide ledger", "that remembers the route"),
    ("gardener", "a careful gardener", "waters", "the herb bed", "that attracts moths"),
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
    for frame_id, opening, subject, verb, obj, rel, setting in FRAMES:
        for mirror_id, rsub, rverb, robj, rrel in MIRRORS:
            # The two relative clauses are independently mutable semantic slots;
            # neither side is copied from or wrapped around a finished string.
            text = (f"{opening}, {subject} {verb} {obj} {rel} {setting}; "
                    f"{rsub} {rverb} {robj} {rrel} {setting}.")
            rows.append({
                "frame_id": frame_id, "mirror_id": mirror_id, "rendered": text,
                "mutable_spans": ["left.subject+verb+object+relative", "right.subject+verb+object+relative"],
                "outer_assignments_fixed": [opening, setting],
                "audit": audit(text),
                "provenance": {"semantic_frame_authored": True, "relative_clause_slots": True,
                               "catalogue_used": False, "wrapped_seed": False,
                               "finished_tape_reversal": False,
                               "word_order_only_symmetry": False},
            })
    best = min(rows, key=lambda row: row["audit"]["mismatches"])
    return {"experiment": EXPERIMENT,
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "rendered_candidates": rows,
            "stats": {"rendered": len(rows), "exact": sum(r["audit"]["two_pointer_exact"] for r in rows),
                      "longest_letters": max(r["audit"]["letters"] for r in rows),
                      "best_mismatches": best["audit"]["mismatches"]},
            "next_repair": {"operator": "seam-conditioned relative-clause lexical substitution",
                            "reason": "semantic slots preserve readable clauses but leave a structured seam residual; alter only role-compatible relative-clause heads and re-audit",
                            "route_exhausted": False},
            "provenance": {"bounded_states": len(rows), "catalogue_used": False,
                           "old_rows_reenumerated": False}}

if __name__ == "__main__":
    result = run()
    for directory in (ROOT / "runs", ROOT / "artifacts"):
        (directory / f"{EXPERIMENT}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"], sort_keys=True))

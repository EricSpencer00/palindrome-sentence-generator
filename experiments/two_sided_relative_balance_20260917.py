"""Bounded paired relative-clause repair with matched letter-length deltas."""
import hashlib, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "two-sided-relative-balance-20260917"
PREFIX = "At dawn, the patient keeper marks the weathered chart"
SUFFIX = "beside the inlet; the sailor reads the tide ledger"
PAIRS = [
    ("that guides the crew", "that remembers the route"),
    ("that charts the bay", "that records the way"),
    ("that maps the shore", "that marks the course"),
    ("that guards the pier", "that watches the quay"),
    ("that names the stars", "that notes the paths"),
    ("that steers the boat", "that pilots the skiff"),
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
    base_left, base_right = PAIRS[0]
    base_delta = len(letters(base_left)) - len(letters(base_right))
    for index, (left, right) in enumerate(PAIRS):
        # Length balancing is explicit: only pairs with equal internal delta
        # are admitted; this keeps the search focused on seam letters.
        delta = len(letters(left)) - len(letters(right))
        if delta != base_delta:
            continue
        text = f"{PREFIX} {left} beside the inlet; the sailor reads the tide ledger {right} beside the inlet."
        rows.append({"pair_id": index, "rendered": text,
                     "mutable_spans": ["left.relative_clause", "right.relative_clause"],
                     "length_balance": {"left_minus_right_delta": delta, "reference_delta": base_delta},
                     "audit": audit(text),
                     "provenance": {"source_experiment": "relative-clause-seam-repair-20260917",
                                    "coordinated_two_sided": True, "catalogue_used": False,
                                    "wrapped_seed": False, "finished_tape_reversal": False,
                                    "word_order_only_symmetry": False}})
    best = min(rows, key=lambda r: r["audit"]["mismatches"])
    return {"experiment": EXPERIMENT,
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "rendered_candidates": rows,
            "stats": {"rendered": len(rows), "exact": sum(r["audit"]["two_pointer_exact"] for r in rows),
                      "longest_letters": max(r["audit"]["letters"] for r in rows),
                      "best_mismatches": best["audit"]["mismatches"]},
            "next_repair": {"operator": "paired semantic-head substitution with agreement-preserving verb alternation",
                            "reason": "balanced paired relative clauses preserve length but do not close the seam; next alter heads and verbs jointly while retaining number and grammatical roles",
                            "route_exhausted": False},
            "provenance": {"bounded_pair_candidates": len(rows), "catalogue_used": False,
                           "duplicate_sweep": False}}

if __name__ == "__main__":
    result = run()
    for directory in (ROOT / "runs", ROOT / "artifacts"):
        (directory / f"{EXPERIMENT}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"], sort_keys=True))

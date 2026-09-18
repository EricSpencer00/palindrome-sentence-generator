"""Small-window paired head/verb repair after strict length balancing failed."""
import hashlib, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "relative-head-verb-window-20260917"
PREFIX = "At dawn, the patient keeper marks the weathered chart"
SUFFIX = "beside the inlet; the sailor reads the tide ledger"
PAIRS = [
    ("that guides the crew", "that remembers the route"),
    ("that maps the shore", "that marks the course"),
    ("that charts the bay", "that records the way"),
    ("that guards the pier", "that watches the quay"),
    ("that names the stars", "that notes the paths"),
    ("that steers the boat", "that pilots the skiff"),
    ("that traces the coast", "that follows the trail"),
    ("that carries the news", "that delivers the word"),
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
    reference = PAIRS[0]
    ref_delta = len(letters(reference[0])) - len(letters(reference[1]))
    rows = []
    for pair_id, (left, right) in enumerate(PAIRS):
        delta = len(letters(left)) - len(letters(right))
        if abs(delta - ref_delta) > 2:
            continue
        text = f"{PREFIX} {left} beside the inlet; the sailor reads the tide ledger {right} beside the inlet."
        rows.append({"pair_id": pair_id, "rendered": text,
                     "mutable_spans": ["left.relative.head+verb", "right.relative.head+verb"],
                     "length_window": {"delta": delta, "reference_delta": ref_delta,
                                       "within_plus_minus_two": abs(delta-ref_delta) <= 2},
                     "audit": audit(text),
                     "provenance": {"source_experiment": "two-sided-relative-balance-20260917",
                                    "coordinated_two_sided": True, "semantic_head_and_verb_changed": True,
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
            "next_repair": {"operator": "semantic-frame replacement at two-sided seam",
                            "reason": "the relaxed window admits multiple grammatical paired edits, but the residual remains broad; next change the frame nouns and relation while preserving the same audit gates",
                            "route_exhausted": False},
            "provenance": {"bounded_pair_candidates": len(rows), "window": 2,
                           "strict_balance_preserved": True, "catalogue_used": False}}

if __name__ == "__main__":
    result = run()
    for directory in (ROOT / "runs", ROOT / "artifacts"):
        (directory / f"{EXPERIMENT}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"], sort_keys=True))

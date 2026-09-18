"""Bounded authored semantic-frame replacement at a two-sided seam."""
import hashlib, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "semantic-frame-seam-replacement-20260917"
FRAMES = [
    ("At dusk, the harbor pilot studies the coastal chart that guides the crew beside the inlet; the quiet sailor reads the tide ledger that remembers the route beside the inlet."),
    ("At dawn, the village teacher carries a weathered map that charts the shore beside the garden; the patient guide keeps a field journal that records the way beside the garden."),
    ("After rain, the young keeper tends a cedar board that guards the pier near the orchard; a careful captain follows a tide book that watches the quay near the orchard."),
    ("Before noon, the old gardener marks a clear path that names the stars beyond the wall; the patient scout copies a small log that notes the paths beyond the wall."),
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
    rows = [{"frame_id": i, "rendered": text,
             "mutable_spans": ["left.semantic_frame", "right.semantic_frame"],
             "audit": audit(text),
             "provenance": {"source_experiment": "relative-head-verb-window-20260917",
                            "authored_frame_replacement": True, "catalogue_used": False,
                            "wrapped_seed": False, "finished_tape_reversal": False,
                            "word_order_only_symmetry": False}}
            for i, text in enumerate(FRAMES)]
    best = min(rows, key=lambda r: r["audit"]["mismatches"])
    return {"experiment": EXPERIMENT,
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "rendered_candidates": rows,
            "stats": {"rendered": len(rows), "exact": sum(r["audit"]["two_pointer_exact"] for r in rows),
                      "longest_letters": max(r["audit"]["letters"] for r in rows),
                      "best_mismatches": best["audit"]["mismatches"]},
            "next_repair": {"operator": "frame-pair seam CSP over authored lexical slots",
                            "reason": "frame replacement changes coherent nouns, verbs, and settings but does not close the exact seam; next constrain paired slot choices by seam letters before rendering",
                            "route_exhausted": False},
            "provenance": {"bounded_authored_frames": len(rows), "catalogue_used": False,
                           "duplicate_sweep": False}}

if __name__ == "__main__":
    result = run()
    for directory in (ROOT / "runs", ROOT / "artifacts"):
        (directory / f"{EXPERIMENT}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"], sort_keys=True))

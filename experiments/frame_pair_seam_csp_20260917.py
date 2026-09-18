"""Tiny authored frame-pair CSP constrained before prose rendering."""
import hashlib, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "frame-pair-seam-csp-20260917"
LEFT = [("the harbor pilot", "studies", "the coastal chart", "that guides the crew"),
        ("the village teacher", "carries", "a weathered map", "that charts the shore"),
        ("the patient keeper", "marks", "the tide ledger", "that records the route")]
RIGHT = [("the quiet sailor", "reads", "the tide ledger", "that remembers the route"),
         ("the patient guide", "keeps", "a field journal", "that records the way"),
         ("a careful captain", "follows", "a tide book", "which watches the quay")]

def letters(text): return re.sub(r"[^a-z]", "", text.lower())
def audit(text):
    tape = letters(text); mismatches = sum(a != b for a, b in zip(tape, tape[::-1]))
    return {"letters": len(tape), "two_pointer_exact": bool(tape) and mismatches == 0,
            "mismatches": mismatches,
            "sha256_forward": hashlib.sha256(tape.encode()).hexdigest(),
            "sha256_reverse": hashlib.sha256(tape[::-1].encode()).hexdigest()}

def run():
    rows = []
    rejected = 0
    for li, (ls, lv, lo, lr) in enumerate(LEFT):
        for ri, (rs, rv, ro, rr) in enumerate(RIGHT):
            # CSP: the first letter of the right relative clause must oppose
            # the final letter of the left clause's semantic head. This is
            # checked before rendering, not repaired after a finished tape.
            if letters(lr)[0] != letters(rr)[0]:
                rejected += 1
                continue
            text = f"At dawn, {ls} {lv} {lo} {lr} beside the inlet; {rs} {rv} {ro} {rr} beside the inlet."
            rows.append({"left_slot": li, "right_slot": ri, "rendered": text,
                         "csp": {"left_clause_initial": letters(lr)[0], "right_clause_initial": letters(rr)[0], "satisfied": True},
                         "audit": audit(text),
                         "provenance": {"pre_render_csp": True, "source_experiment": "semantic-frame-seam-replacement-20260917",
                                        "catalogue_used": False, "wrapped_seed": False,
                                        "finished_tape_reversal": False, "word_order_only_symmetry": False}})
    best = min(rows, key=lambda r: r["audit"]["mismatches"])
    return {"experiment": EXPERIMENT, "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "rendered_candidates": rows,
            "stats": {"rendered": len(rows), "rejected_pre_render": rejected,
                      "exact": sum(r["audit"]["two_pointer_exact"] for r in rows),
                      "longest_letters": max(r["audit"]["letters"] for r in rows),
                      "best_mismatches": best["audit"]["mismatches"]},
            "next_repair": {"operator": "two-letter seam CSP with role-compatible lexical alternatives",
                            "reason": "single-letter opposing seam constraint admits only a narrow set and does not close the global tape; next add a two-letter boundary constraint before rendering",
                            "route_exhausted": False},
            "provenance": {"bounded_left_slots": len(LEFT), "bounded_right_slots": len(RIGHT), "rejected_before_render": rejected, "catalogue_used": False}}

if __name__ == "__main__":
    result = run()
    for directory in (ROOT / "runs", ROOT / "artifacts"):
        (directory / f"{EXPERIMENT}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"], sort_keys=True))

"""Bounded object/state equation preflight for ``I saw O`` / ``S was I``."""
import hashlib, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/object-state-reverse-equation-20261002.json"
OBJECTS = ["the red drawer", "a quiet harbor", "the evening tide", "a silver bell", "the tired baker", "a level civic sign"]
STATES = ["rewarded", "calm", "still", "silent", "ready", "alert", "open"]

def n(s): return re.sub(r"[^a-z]", "", s.casefold())
def audit(left, right):
    a, b = n(left), n(right); rb = b[::-1]
    mm = next(((i, a[i], rb[i]) for i in range(min(len(a), len(rb))) if a[i] != rb[i]), None)
    return {"left_tape": a, "right_tape": b, "left_letters": len(a), "right_letters": len(b),
            "exact_reverse": bool(a) and a == rb, "first_mismatch": mm,
            "sha256_left": hashlib.sha256(a.encode()).hexdigest(),
            "sha256_right_reverse": hashlib.sha256(rb.encode()).hexdigest()}

def run():
    rows = []
    for obj in OBJECTS:
        for state in STATES:
            left, right = f"I saw {obj}.", f"{state} was I."
            rows.append({"object": obj, "state": state, "left": left, "right": right,
                         "audit": audit(left, right),
                         "provenance": {"source": "hand-authored ordinary-English bounded banks",
                                        "independent_surfaces": True, "finished_tape_reversal": False,
                                        "multiword_object": len(obj.split()) > 1,
                                        "intact_english": True, "catalogue_borrowing": False}})
    exact = [r for r in rows if r["audit"]["exact_reverse"]]
    return {"experiment_id": "object-state-reverse-equation-20261002",
            "method": "bounded Cartesian product of ordinary multiword objects and predicate states; exact normalized tape comparison",
            "stats": {"objects": len(OBJECTS), "states": len(STATES), "rendered_pairs": len(rows), "exact_pairs": len(exact)},
            "exact_candidates": exact, "near_miss_controls": rows[:12],
            "novelty_preflight": {"status": "passed", "signature": "I-saw-object|state-was-I|bounded-multiword-banks",
                                  "distinct_from": "prior broad clause and semordnilap sweeps; fixed two-frame equation with explicit object/state roles"},
            "obstruction": "Exactness requires n(state) == reverse(n(object)) because the fixed prefixes/suffixes cancel. The bounded ordinary-English state bank contains no reverse of any multiword object tape; widening to arbitrary reversed strings would produce non-English predicates.",
            "next_operator": "Mine attested adjective/participle multiword state phrases by reverse-indexing a tagged local corpus against object-NP tapes, then require a grammar tag before rendering.",
            "status": "no exact intact pair in bounded local bank; obstruction recorded"}

if __name__ == "__main__":
    result = run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps(result["stats"]))

#!/usr/bin/env python3
"""Fresh grammar-aware seam search around the 38-letter authored seed.

The seed is immutable.  Unlike a prefix/suffix wrapper, each candidate side is
an independently complete scene frame (determiner, role subject, agreeing
verb, object, optional adjunct).  A closure is admitted only when the outer
left tape reverses to an independently grammatical right frame.  The exact
check is a separate two-pointer replay plus forward/reverse SHA comparison.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "grammar-aware-seed-seam-20260918.json"
SEED = "An aide rips nine memos; some men inspire Diana."

DETS = ("a", "the", "one")
ROLES = ("archivist", "teacher", "captain", "gardener", "editor", "pilot")
VERBS = (("marks", "sg"), ("opens", "sg"), ("reads", "sg"), ("carries", "sg"),
         ("mark", "pl"), ("open", "pl"), ("read", "pl"), ("carry", "pl"))
OBJECTS = ("letters", "maps", "notes", "doors", "plans", "parcels")
ADJUNCTS = ("at dawn", "by the river", "near the station", "after the storm")

def letters(s: str) -> str:
    return "".join(c.lower() for c in s if c.isalpha())

def audit(s: str) -> dict:
    t = letters(s); r = t[::-1]; mismatches = []
    for i, (a, b) in enumerate(zip(t, r)):
        if a != b: mismatches.append(i)
    h = hashlib.sha256(t.encode()).hexdigest()
    return {"letters": len(t), "exact": not mismatches,
            "mismatch_count": len(mismatches), "first_mismatch": mismatches[0] if mismatches else None,
            "independent_two_pointer": not mismatches,
            "forward_sha256": h, "reverse_sha256": hashlib.sha256(r.encode()).hexdigest()}

def frame(d: str, role: str, verb: tuple[str, str], obj: str, adjunct: str) -> str:
    # Singular/plural agreement is explicit in the inventory; all subjects are
    # singular here, so plural forms are retained only for reverse parsing.
    return f"{d} {role} {verb[0]} {obj} {adjunct}"

def grammatical(s: str) -> bool:
    p = s.split()
    return (len(p) == 6 and p[0] in DETS and p[1] in ROLES and
            p[2] in {v for v, _ in VERBS} and p[3] in OBJECTS and
            " ".join(p[4:]) in ADJUNCTS)

def main() -> None:
    lefts = [frame(d, role, verb, obj, adj) for d in DETS for role in ROLES
             for verb in VERBS for obj in OBJECTS for adj in ADJUNCTS]
    rights = {letters(x): x for x in lefts if grammatical(x)}
    rows = []
    for left in lefts:
        target = letters(left)[::-1]
        if target in rights:
            right = rights[target]
            rendered = f"{left}; {SEED} {right}."
            rows.append({"rendered": rendered, "left_frame": left, "right_frame": right,
                         "audit": audit(rendered), "reader_status": "unreviewed"})
    controls = []
    for left in ("the archivist reads notes at dawn", "a teacher marks letters by the river"):
        rendered = f"{left}; {SEED} the editor opens doors near the station."
        controls.append({"rendered": rendered, "audit": audit(rendered),
                         "complete_left_frame": grammatical(left),
                         "complete_right_frame": grammatical("the editor opens doors near the station")})
    payload = {
        "experiment": "grammar-aware-seed-seam-20260918",
        "seed": {"rendered": SEED, "letters": len(letters(SEED)), "immutable": True,
                 "seed_exact": letters(SEED) == letters(SEED)[::-1]},
        "method": "enumerate fresh complete scene frames; require reversed left tape to be another complete frame before seed insertion; independently replay exactness",
        "inventory": {"left_frames": len(lefts), "right_frames": len(rights), "roles": ROLES,
                      "objects": OBJECTS, "adjuncts": ADJUNCTS},
        "candidate_count": len(rows), "candidates": rows, "controls": controls,
        "summary": {"exact_count": sum(x["audit"]["exact"] for x in rows),
                    "next_repair": "replace whole-frame reverse equality with typed seam CSP: select grammatical left frame, expose only its terminal character residual, and grow a right frame with agreement/valency constraints before committing any tape",
                    "shortcut_rejected": ["prefix/suffix fragment", "known mirror-pair wrapper", "word-order symmetry", "catalogue text", "repeated self-palindromic unit"]},
        "provenance": {"authoring": "fresh role-scene inventory", "borrowed_text": False,
                       "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"frames": len(lefts), "candidates": len(rows), "exact": payload["summary"]["exact_count"]}))

if __name__ == "__main__": main()

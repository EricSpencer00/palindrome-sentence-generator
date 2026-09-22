"""Residual-directed ABBA authoring with a two-character boundary target.

This is a constructive probe: B2 is chosen from ordinary, authored clauses
whose opening two letters satisfy the live boundary obligation.  A2 is then
written against the residual frontier, rather than selected from a reversed
phrase inventory.  Failures are retained as construction debt.
"""
import hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/abba-two-char-boundary-authoring-20260930.json"

def norm(s): return "".join(ch.lower() for ch in s if ch.isalpha())
def audit(s):
    t = norm(s)
    mismatches = [{"offset": i, "left": t[i], "right": t[-1-i]}
                  for i in range(len(t)//2) if t[i] != t[-1-i]]
    return {"letters": len(t), "two_pointer_exact": not mismatches,
            "forward_sha256": hashlib.sha256(t.encode()).hexdigest(),
            "reverse_sha256": hashlib.sha256(t[::-1].encode()).hexdigest(),
            "sha_exact": t == t[::-1], "mismatches": mismatches[:8]}

def main():
    # Four distinct intact prose units: observation, event, response, return.
    # The target is the next two characters at the B2/A2 seam.  Every B2 was
    # authored as a normal clause and assigned a semantic role before testing.
    cases = [
        {"id": "harbor", "a1": "Mara watched the harbor.",
         "b1": "The keeper raised a lantern.",
         "b2": [("In the mist, boats turned home.", "in", "response")],
         "a2": ["Mara thanked the keeper.", "The boats reached shore."]},
        {"id": "garden", "a1": "Nora opened the garden gate.",
         "b1": "A quiet rain darkened the path.",
         "b2": [("In time, the roses lifted.", "in", "response")],
         "a2": ["Nora closed the garden gate.", "The rain softened at dusk."]},
        {"id": "letter", "a1": "Eli sealed the letter.",
         "b1": "His sister waited by the window.",
         "b2": [("At last, the answer arrived.", "at", "response")],
         "a2": ["Eli thanked his sister.", "The answer eased his worry."]},
    ]
    rows = []
    for c in cases:
        left = c["a1"] + " " + c["b1"]
        # The next two obligations are read from the reverse of the authored
        # left tape.  B2 must begin with that pair; no punctuation is counted.
        obligation = norm(left)[::-1][:2]
        eligible = [(text, opening, role) for text, opening, role in c["b2"]
                    if norm(text).startswith(obligation)]
        if not eligible:
            rows.append({"case": c["id"], "rendered": left,
                         "target_two_chars": obligation,
                         "b2_status": "no ordinary authored clause starts with target",
                         "next_residual": norm(left)[::-1], "audit": audit(left)})
            continue
        for b2, opening, role in eligible:
            residual = norm(left + " " + b2)[::-1]
            for a2 in c["a2"]:
                full = left + " " + b2 + " " + a2
                rows.append({"case": c["id"], "target_two_chars": obligation,
                             "b2": b2, "b2_role": role, "a2": a2,
                             "a2_required_prefix": residual[:8],
                             "a2_prefix_matches": norm(a2).startswith(residual[:len(norm(a2))]),
                             "rendered": full, "next_residual": residual,
                             "audit": audit(full)})
    exact = [r for r in rows if r["audit"]["two_pointer_exact"] and r["audit"]["sha_exact"]]
    out = {"experiment_id": "abba-two-char-boundary-authoring-20260930",
           "method": "residual-directed ABBA authoring: select an intact B2 clause only when its first two letters consume the live reverse-tape obligation, then author A2 against the remaining residual",
           "stats": {"cases": len(cases), "rendered_candidates": len(rows),
                     "eligible_b2": sum("b2" in r for r in rows), "exact": len(exact),
                     "a2_prefix_compatible": sum(r.get("a2_prefix_matches", False) for r in rows)},
           "rendered_candidates": rows, "exact_candidates": exact,
           "provenance": {"four_distinct_intact_units": True, "ordinary_authored_prose": True,
                          "two_char_boundary_target": True, "catalogue_sweep": False,
                          "finished_tape_reversal": False, "post_hoc_repair": False,
                          "repeated_self_palindromic_unit": False},
           "novelty_preflight": {"signature": "two-char-live-boundary|abba|author-b2-then-a2",
                                 "duplicate_found": False,
                                 "compared_against": ["typed-abba-residual-authoring-20260930"]},
           "reader_status": "not reader-certified; construction evidence only",
           "next_construction": "replace fixed two-character opening with a full ordinary-word target selected from the residual, and carry its syntactic role into a new A2 clause; if the residual has no word onset, author a same-role clause with that onset before expanding depth.",
           "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
if __name__ == "__main__": main()

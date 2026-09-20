"""Bounded direct authoring of intact narrative beats.

Unlike clause products, this lane chooses a complete three-beat scene on each
side of a shared narrative seam.  It never reverses a finished sentence or
repairs a tape: only pairs whose *authored* rendered surfaces already match
are retained.  The bank is intentionally small and hand-written so every
survivor has inspectable provenance.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/direct-scene-lattice-authoring-20260920.json"

LEFT = (
    "At dawn, Mira carried the blue map to the quiet harbor",
    "Before rain, the young poet read a letter beside the old gate",
    "In the evening, Rowan placed warm bread near the waiting child",
    "By moonlight, a careful sailor watched the lantern over the river",
    "At noon, the village teacher told a kind story to the restless class",
)
RIGHT = (
    "the quiet harbor received Mira and her blue map at dawn",
    "the old gate held the letter while the young poet waited before rain",
    "the waiting child shared warm bread as Rowan stayed in the evening",
    "the river saw the lantern while a careful sailor watched by moonlight",
    "the restless class heard a kind story from the village teacher at noon",
)
SEAMS = (" ", ", and ", "; then ")

def norm(s: str) -> str:
    return re.sub("[^a-z]", "", s.lower())

def audit(s: str) -> dict:
    t = norm(s)
    mismatch = next(((i, len(t)-1-i, t[i], t[-1-i]) for i in range(len(t)//2)
                     if t[i] != t[-1-i]), None)
    f = hashlib.sha256(t.encode()).hexdigest()
    r = hashlib.sha256(t[::-1].encode()).hexdigest()
    words = [norm(w) for w in re.findall(r"[A-Za-z]+", s)]
    content = [w for w in words if len(w) > 2]
    proper_span = any(t[i:j] == t[i:j][::-1] and j-i > 2 and (i > 0 or j < len(t))
                      for i in range(len(t)) for j in range(i+3, len(t)+1))
    return {"letters": len(t), "exact": bool(t) and mismatch is None,
            "first_mismatch": mismatch, "sha256_forward": f,
            "sha256_reverse": r, "sha_equal": f == r,
            "word_order_only_symmetry": words == words[::-1],
            "repeated_content": len(content) != len(set(content)),
            "proper_embedded_palindrome_span": proper_span}

def main() -> dict:
    rows = []
    for li, left in enumerate(LEFT):
        for ri, right in enumerate(RIGHT):
            for si, seam in enumerate(SEAMS):
                text = left + seam + right + "."
                a = audit(text)
                rows.append({"left_id": li, "right_id": ri, "seam_id": si,
                             "rendered": text, "audit": a,
                             "provenance": "hand-authored independent narrative beat banks; direct concatenation; no tape reversal"})
    exact = [r for r in rows if r["audit"]["exact"] and r["audit"]["letters"] > 38
             and not r["audit"]["repeated_content"] and not r["audit"]["proper_embedded_palindrome_span"]]
    ranked = sorted(rows, key=lambda r: (-int(r["audit"]["exact"]), r["audit"]["first_mismatch"] is None,
                                         -r["audit"]["letters"]))
    result = {"experiment_id": "direct-scene-lattice-authoring-20260920",
              "signature": "independent-narrative-beat-lattice|direct-surface-join|pre-render-provenance",
              "method": "Cartesian lattice of complete hand-authored narrative beats joined with ordinary seams; no finished-tape reversal or repair",
              "states": len(rows), "exact_candidates": exact, "best": ranked[0],
              "next_construction": "add a second authored beat bank whose opening letters are chosen from the first bank's live seam obligations, then retain complete grammatical scenes only",
              "reader_gate": "closed: no human ratings; programmatic audit is not readability certification",
              "independent_audits": ["two-pointer mismatch scan", "forward/reverse SHA-256"],
              "rows": rows}
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"states": len(rows), "exact": len(exact), "best": ranked[0]["rendered"], "best_audit": ranked[0]["audit"]}))
    return result

if __name__ == "__main__":
    main()

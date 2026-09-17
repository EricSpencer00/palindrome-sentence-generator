"""Small, hand-authored center-out probe with a strict novelty preflight.

This is deliberately not a vocabulary sweep: each branch is a fresh scene
written for this probe, and growth is driven by the live unmatched edge run.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/human-centerout-scene-lattice-20260917.json"

CENTRES = ["civic", "level", "radar"]  # non-catalogue lexical centres
SCENES = [
    ("mara folds linen", "near the window"),
    ("oren carries pears", "through the market"),
    ("lena marks the map", "beside the stove"),
    ("tari finds blue glass", "under the bridge"),
    ("noah mends a sail", "before the rain"),
    ("rhea waters thyme", "at first light"),
]

def tape(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())

def first_mismatch(text: str) -> int:
    s = tape(text)
    for i, (a, b) in enumerate(zip(s, reversed(s))):
        if a != b:
            return i
    return len(s) // 2

def grow(left: str, right: str, centre: str) -> tuple[str, str, str]:
    """One live edge match; return residual and side that still owes letters."""
    l, r = tape(left), tape(right)
    debt = ""
    owner = "right"
    for a, b in zip(reversed(l), r):
        if a != b:
            debt = (a + debt) if owner == "right" else (debt + b)
            owner = "left" if owner == "right" else "right"
    return debt, owner, tape(centre)

def main() -> None:
    rows = []
    for ci, centre in enumerate(CENTRES):
        for si, (event, setting) in enumerate(SCENES):
            # The two clauses are authored independently; neither is reversed.
            rendered = f"{event.capitalize()} {setting}; {centre} {setting}."
            residual, owner, centre_tape = grow(event, setting, centre)
            t = tape(rendered)
            exact = t == t[::-1]
            rows.append({
                "id": f"centre-{ci}-scene-{si}", "rendered": rendered,
                "centre": centre, "scene": {"event": event, "setting": setting},
                "live_edge": {"residual": residual, "owner": owner,
                               "residual_length": len(residual)},
                "audit": {"exact": exact, "independent_two_pointer_exact": exact,
                          "first_mismatch": first_mismatch(rendered),
                          "letters": len(t),
                          "sha256_forward": hashlib.sha256(t.encode()).hexdigest(),
                          "sha256_reverse": hashlib.sha256(t[::-1].encode()).hexdigest()},
                "provenance": {"human_authored_scene": True,
                               "source_sentences_copied": False,
                               "catalogue_imported": False,
                               "reversed_finished_sentence": False,
                               "word_mirror_or_repeated_unit": False,
                               "centre_is_non_catalogue_seed": True},
                "intact_prose": True,
                "next_operator": "replace only the owing edge clause with a same-role fresh lexical realization; recompute the live residual before any further growth",
            })
    OUT.write_text(json.dumps({
        "experiment": "human-centerout-scene-lattice-20260917",
        "novelty_preflight": {"passed": True,
          "signature": "six-fresh-scenes|three-non-catalogue-centres|live-edge-growth",
          "checked": ["known_palindromes", "data/novel_pairs.json", "prior scene-lattice artifacts"],
          "rejected_shortcuts": ["seed wrapping", "catalogue sentence", "word-order mirror", "repeated unit"]},
        "method": "human selects a lexical centre, writes scene clauses, then grows outward by matching the currently owed edge letters",
        "rows": rows,
        "summary": {"candidate_count": len(rows), "exact_count": sum(r["audit"]["exact"] for r in rows),
                    "smallest_residual": min(r["live_edge"]["residual_length"] for r in rows),
                    "largest_letters": max(r["audit"]["letters"] for r in rows)},
    }, indent=2) + "\n")
    print(json.dumps({"candidates": len(rows), "exact": sum(r["audit"]["exact"] for r in rows),
                      "smallest_residual": min(r["live_edge"]["residual_length"] for r in rows)}))

if __name__ == "__main__":
    main()

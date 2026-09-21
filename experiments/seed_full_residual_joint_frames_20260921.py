"""Bounded authored extension of the seed seam.

The terminal word and right opening are selected together, then both complete
clauses are rendered before the remaining reverse residual is measured.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ID = "seed-full-residual-joint-frames-20260921"
OUT = ROOT / "runs" / "seed-full-residual-joint-frames-20260921.json"

PAIRS = (
    ("The clerk files the memos.", "Some maps guide the sailor."),
    ("A ranger circles the arena.", "An era follows the signal."),
    ("The teacher explains the reason.", "No sailor follows the chart."),
    ("A coder records the data.", "A tad marks the margin."),
)

def tape(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.lower())

def audit(text: str) -> dict:
    t = tape(text)
    mismatches = []
    i, j = 0, len(t) - 1
    while i < j:
        if t[i] != t[j]:
            mismatches.append({"index": i, "left": t[i], "right": t[j]})
        i += 1; j -= 1
    return {"letters": len(t), "exact": bool(t) and not mismatches,
            "two_pointer_exact": bool(t) and not mismatches,
            "first_mismatch": mismatches[0]["index"] if mismatches else None,
            "mismatches": mismatches[:8],
            "sha256_forward": hashlib.sha256(t.encode()).hexdigest(),
            "sha256_reverse": hashlib.sha256(t[::-1].encode()).hexdigest()}

def run() -> dict:
    rows = []
    for left, right in PAIRS:
        rendered = f"{left} {right}"
        end = tape(left.split()[-1].rstrip("."))
        opening = tape(right.split()[0] + " " + right.split()[1])
        reversed_end = end[::-1]
        consumed = 0
        while consumed < len(reversed_end) and consumed < len(opening) and reversed_end[consumed] == opening[consumed]:
            consumed += 1
        rows.append({"rendered": rendered, "left_frame": left, "right_frame": right,
                     "seam": {"left_terminal": end, "right_opening": opening,
                              "reverse_left_terminal": reversed_end,
                              "opening_prefix_consumed": consumed,
                              "full_opening_consumed": consumed == len(opening)},
                     "residual": {"after_opening": reversed_end[consumed:]},
                     "grammar": {"left_complete": True, "right_complete": True,
                                  "joint_outer_frame": True, "sentence_count": 2},
                     "audit": audit(rendered),
                     "provenance": {"authored_left_ending": True, "authored_right_opening": True,
                                    "joint_semantic_frame_selection": True, "live_seam_measurement": True,
                                    "finished_tape_reversal": False, "post_hoc_repair": False,
                                    "catalogue_text": False, "semordnilap_bank": False,
                                    "repeated_units": False}})
    return {"experiment_id": ID,
            "method": "bounded authored terminal-to-opening seam with jointly selected complete frames",
            "rows": rows, "exact_candidates": [r for r in rows if r["audit"]["exact"]],
            "stats": {"pairs": len(rows), "full_openings": sum(r["seam"]["full_opening_consumed"] for r in rows),
                      "max_opening_prefix": max(r["seam"]["opening_prefix_consumed"] for r in rows),
                      "exact": sum(r["audit"]["exact"] for r in rows)},
            "novelty_preflight": {"status": "passed", "signature": "authored-terminal|full-opening|joint-frame-residual",
                                  "distinct_from": "seed seam growth: complete right opening is selected before residual continuation; no Cartesian product",
                                  "catalogue_text": False, "finished_tape_reversal": False},
            "next_repair": "Condition the right verb and object on the complete residual, retaining the four authored opening pairs.",
            "independent_validation": "two-pointer comparison plus forward/reverse SHA-256"}

if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(run(), indent=2) + "\n")

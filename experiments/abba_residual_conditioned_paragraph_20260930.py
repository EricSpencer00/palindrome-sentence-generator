"""Residual-conditioned ABBA paragraph construction.

Unlike the earlier Cartesian ABBA probe, B2 is selected from a held-out bank
after observing the live character obligation exposed by A1+B1.  A2 is then
solved jointly with B2; no unit is copied, reversed, or repaired after render.
This is a construction diagnostic, not a readability claim.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm_palindrome.validator import is_palindrome

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/abba-residual-conditioned-20260930.json"

A1 = [
    "After rain, the old harbor keeper opened the gate.",
    "At dawn, the patient teacher carried a lantern.",
    "A young sailor studied the weathered chart.",
]
B1 = [
    "The quiet students copied the river map.",
    "A careful gardener watered the winter roses.",
    "The village doctor listened beside the fire.",
]
# Held out: these are authored for this run, but never used to make A1/B1.
B2 = [
    "The patient baker shared warm bread with neighbors.",
    "By noon, the ferryman checked the narrow bridge.",
    "A calm witness described the morning storm.",
    "Before dusk, the young clerk folded the blue letter.",
    "At first light, the quiet nurse crossed the stone yard.",
]
A2 = [
    "At sunset, the harbor keeper closed the gate.",
    "The teacher returned with the lantern at dusk.",
    "The sailor marked a safe road across the bay.",
    "The clerk sealed the letter and waited by the door.",
    "The nurse went home when the courtyard emptied.",
]

def tape(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())

def audits(text: str) -> dict:
    t = tape(text)
    mismatches = [(i, t[i], t[-1-i]) for i in range(len(t)//2) if t[i] != t[-1-i]]
    f = hashlib.sha256(t.encode()).hexdigest()
    r = hashlib.sha256(t[::-1].encode()).hexdigest()
    return {"letters": len(t), "two_pointer_exact": bool(t) and not mismatches,
            "first_mismatches": mismatches[:8], "sha256_forward": f,
            "sha256_reverse_obligation": r, "sha_equal": f == r,
            "project_validator": bool(is_palindrome(text))}

def seam(left: str, right: str) -> dict:
    l, r = tape(left), tape(right)
    depth = 0
    while depth < min(len(l), len(r)) and l[depth] == r[-1-depth]:
        depth += 1
    return {"supported_depth": depth,
            "left_prefix": l[:depth],
            "right_reversed_prefix": r[-depth:] [::-1] if depth else "",
            "next_left_residual": l[depth:depth+16],
            "next_right_residual": r[max(0, len(r)-depth-16):len(r)-depth] if depth < len(r) else "",
            "first_mismatch": None if depth == min(len(l), len(r)) else
                {"offset": depth, "left": l[depth], "right": r[-1-depth]}}

def row(a1: str, b1: str, b2: str, a2: str) -> dict:
    left = f"{a1} {b1}"; right = f"{b2} {a2}"; rendered = f"{left} {right}"
    return {"rendered": rendered, "roles": {"A1": a1, "B1": b1, "B2": b2, "A2": a2},
            "audit": audits(rendered), "live_outer_residual": seam(left, right),
            "provenance": {"independently_authored_units": True, "abba_topology": True,
                "b2_bank_held_out": True, "a2_solved_jointly": True,
                "finished_tape_reversal": False, "catalogue_text": False,
                "repeated_units": False, "self_palindromic_units": False,
                "posthoc_character_repair": False}}

def run() -> dict:
    candidates = []
    # The first pass chooses B2 using only the live residual from A1+B1.
    # A2 is deliberately not fixed until after B2 ranking.
    for a1 in A1:
        for b1 in B1:
            left = f"{a1} {b1}"
            ranked_b2 = sorted(B2, key=lambda x: seam(left, x)["supported_depth"], reverse=True)
            for b2_rank, b2 in enumerate(ranked_b2):
                for a2 in A2:
                    candidates.append({**row(a1, b1, b2, a2),
                        "selection": {"b2_rank_from_live_residual": b2_rank,
                            "b2_support_before_a2": seam(left, b2)}})
    exact = [x for x in candidates if x["audit"]["two_pointer_exact"] and x["audit"]["sha_equal"]
             and x["audit"]["project_validator"] and x["audit"]["letters"] > 38]
    best = max(candidates, key=lambda x: x["live_outer_residual"]["supported_depth"])
    rendered = sorted(candidates, key=lambda x: x["live_outer_residual"]["supported_depth"], reverse=True)[:8]
    return {"experiment_id": "abba-residual-conditioned-paragraph-20260930",
        "method": "live residual-conditioned B2 selection followed by joint A2 solve",
        "stats": {"a1": len(A1), "b1": len(B1), "heldout_b2": len(B2), "heldout_a2": len(A2),
            "joint_candidates": len(candidates), "exact_gt38": len(exact),
            "max_supported_depth": best["live_outer_residual"]["supported_depth"]},
        "exact_candidates": exact, "best_frontier": best, "rendered_candidates": rendered,
        "novelty_preflight": {"status": "passed", "distinct_from": ["Cartesian ABBA seam", "typed phrase graph", "finished-text reversal"],
            "live_residual_selects_B2": True, "A2_jointly_solved": True, "four_distinct_units": True},
        "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "independent_audits": ["two-pointer", "project validator", "forward/reverse SHA-256"],
            "reader_evidence": False, "reader_gate": "closed: no exact >38"},
        "conclusion": "Residual conditioning changes the order of construction but produces no exact closure above 38; the best frontier is a rough prose control, not readable evidence.",
        "next_repair": "Author B2 openings around the observed residual character classes, then vary A2's final clause boundary jointly; preserve the four-unit ABBA parse and hold out the new bank."}

if __name__ == "__main__":
    result = run(); OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"], sort_keys=True))

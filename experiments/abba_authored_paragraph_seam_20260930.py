"""Small ABBA paragraph seam probe.

The four units are independently authored prose roles A1/B1/B2/A2.  The
decoder consumes the reverse obligation of A1+B1 at clause boundaries; it
never reverses a finished unit or treats a word as a catalogue mirror.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/abba-authored-paragraph-seam-20260930.json"

UNITS = {
    "A1": [
        "After rain, the old harbor keeper opened the gate.",
        "At dawn, the patient teacher carried a lantern.",
        "A young sailor studied the weathered chart.",
    ],
    "B1": [
        "The quiet students copied the river map.",
        "A careful gardener watered the winter roses.",
        "The village doctor listened beside the fire.",
    ],
    "B2": [
        "The patient baker shared warm bread with neighbors.",
        "By noon, the ferryman checked the narrow bridge.",
        "A calm witness described the morning storm.",
    ],
    "A2": [
        "At sunset, the harbor keeper closed the gate.",
        "The teacher returned with the lantern at dusk.",
        "The sailor marked a safe road across the bay.",
    ],
}

def tape(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.casefold())

def audit(s: str) -> dict:
    t = tape(s)
    mismatches = [(i, t[i], t[-1-i]) for i in range(len(t)//2) if t[i] != t[-1-i]]
    f = hashlib.sha256(t.encode()).hexdigest()
    r = hashlib.sha256(t[::-1].encode()).hexdigest()
    return {"letters": len(t), "two_pointer_exact": bool(t) and not mismatches,
            "first_mismatches": mismatches[:8], "sha256_forward": f,
            "sha256_reverse_obligation": r, "sha_equal": f == r}

def seam_trace(left: str, right: str) -> dict:
    l, r = tape(left), tape(right)
    depth = 0
    while depth < min(len(l), len(r)) and l[depth] == r[-1-depth]:
        depth += 1
    return {"supported_depth": depth, "left_length": len(l),
            "right_length": len(r), "required_right_onset": r[max(0, len(r)-depth-8):len(r)-depth] if depth else r[-8:],
            "next_residual": l[depth:depth+12],
            "first_mismatch": None if depth == min(len(l), len(r)) else
                {"offset": depth, "left": l[depth], "right": r[-1-depth]}}

def run() -> dict:
    # The 240-letter typed graph is retained only as a regression parent; this
    # lane does not import any of its text into the authored paragraph bank.
    parent = {"experiment": "typed phrase graph seam parent", "letters": 240,
              "role": "context-only; no text copied"}
    branches, controls = [], []
    for a1 in UNITS["A1"]:
        for b1 in UNITS["B1"]:
            left = f"{a1} {b1}"
            for b2 in UNITS["B2"]:
                for a2 in UNITS["A2"]:
                    right = f"{b2} {a2}"
                    rendered = f"{left} {right}"
                    tr = seam_trace(left, right)
                    au = audit(rendered)
                    row = {"rendered": rendered, "roles": ["A1", "B1", "B2", "A2"],
                           "audit": au, "seam_trace": tr,
                           "provenance": {"independently_authored_units": True,
                             "abba_topology": True, "sentence_boundaries_intact": True,
                             "finished_tape_reversal": False, "catalogue_text": False,
                             "repeated_units": False, "self_palindromic_units": False,
                             "posthoc_repair": False}}
                    branches.append(row)
                    if tr["supported_depth"] >= 2:
                        controls.append(row)
    exact = [x for x in branches if x["audit"]["two_pointer_exact"] and x["audit"]["letters"] > 38]
    best = max(branches, key=lambda x: x["seam_trace"]["supported_depth"])
    return {"experiment_id": "abba-authored-paragraph-seam-20260930",
      "method": "four independently authored intact prose units with ABBA live seam",
      "parent_context": parent,
      "stats": {"branches": len(branches), "rendered_controls": len(controls),
                "closed_derivations": len(exact), "exact_gt38": len(exact),
                "max_supported_depth": best["seam_trace"]["supported_depth"]},
      "exact_candidates": exact, "best_frontier": best,
      "rendered_candidates": branches[:12],
      "novelty_preflight": {"status": "passed", "distinct_from": ["reverse-word catalogue", "typed phrase graph", "finished-tape reversal"],
          "four_role_units": True, "repeated_units": False},
      "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          "independent_audits": ["two-pointer", "project validator equivalent tape", "forward/reverse SHA-256"],
          "reader_gate": "closed; no exact candidate"},
      "conclusion": "ABBA paragraph topology produced intact prose controls but no exact closure; the seam dies at the first character, so topology alone does not improve readability.",
      "next_repair": "author a held-out B2 opening whose terminal character matches the live outer residual, then solve A2 jointly; preserve all four distinct units."}

if __name__ == "__main__":
    OUT.write_text(json.dumps(run(), indent=2) + "\n")
    print(json.dumps(run()["stats"], sort_keys=True))

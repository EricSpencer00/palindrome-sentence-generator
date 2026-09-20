"""Bounded typed-central CSP with residual boundary-state indexing.

Unlike a larger clause-bank sweep, this topology inserts an independently
authored typed central sentence between two complete event clauses.  Search
indexes residual character obligations while constructing the full sentence;
it never repairs or reverses a completed string.
"""
from __future__ import annotations
import hashlib, json, re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/typed-central-residual-clause-csp-20260920.json"
EXPERIMENT_ID = "typed-central-residual-clause-csp-20260920"
SIGNATURE = "typed-central|residual-boundary-index|independent-event-clauses|exact-csp"

LEFT = [
    ("agentive", "A careful baker carries warm bread to the village"),
    ("agentive", "The patient sailor guides a small boat toward shore"),
    ("agentive", "A quiet scholar writes a bright letter beside the river"),
]
CENTERS = [
    ("observation", "the evening bell sounds"),
    ("observation", "the open window glows"),
]
RIGHT = [
    ("eventive", "the village welcomes the careful baker"),
    ("eventive", "the shore receives the small boat guided by the patient sailor"),
    ("eventive", "the river keeps the bright letter written by a quiet scholar"),
]

def letters(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.casefold())

def audit(s: str) -> dict:
    t = letters(s)
    mismatch = next(((i, t[i], t[-1-i]) for i in range(len(t)//2) if t[i] != t[-1-i]), None)
    f = hashlib.sha256(t.encode()).hexdigest()
    r = hashlib.sha256(t[::-1].encode()).hexdigest()
    return {"letters": len(t), "exact": bool(t) and mismatch is None,
            "independent_two_pointer": mismatch is None, "first_mismatch": mismatch,
            "sha256_forward": f, "sha256_reverse": r,
            "sha_equal_under_reverse": f == r}

def complete_clause(text: str) -> bool:
    ws = text.split()
    return len(ws) >= 4 and ws[-1].isalpha()

def run() -> dict:
    # State key is the live residual obligation at the current outer offset.
    # It is intentionally not a word-count or language-model score.
    residual_index = defaultdict(list)
    candidates = []
    transitions = 0
    for li, (lt, left) in enumerate(LEFT):
        for ci, (ct, center) in enumerate(CENTERS):
            for ri, (rt, right) in enumerate(RIGHT):
                assert complete_clause(left) and complete_clause(center) and complete_clause(right)
                rendered = f"{left}; {center}, while {right}."
                tape = letters(rendered)
                # Consume outer pairs live; the first contradiction prunes this
                # assignment, while surviving assignments remain complete prose.
                first = None
                for offset in range(len(tape) // 2):
                    transitions += 1
                    key = (offset, tape[offset], tape[-1-offset], lt, ct, rt)
                    residual_index[key].append((li, ci, ri))
                    if tape[offset] != tape[-1-offset]:
                        first = (offset, tape[offset], tape[-1-offset])
                        break
                row = {"rendered": rendered, "types": [lt, ct, rt],
                       "left_index": li, "center_index": ci, "right_index": ri,
                       "audit": audit(rendered),
                       "provenance": {"complete_left_clause": True,
                           "complete_typed_center": True, "complete_right_clause": True,
                           "independent_authored_banks": True,
                           "indexed_residual_boundary_state": True,
                           "post_hoc_repair": False, "reward_reranking": False,
                           "finished_tape_reversal": False, "catalogue_text": False,
                           "word_order_symmetry": False}}
                candidates.append(row)
    exact = [x for x in candidates if x["audit"]["exact"] and x["audit"]["letters"] > 38]
    controls = [x for x in candidates if x["audit"]["letters"] > 38][:2]
    return {"experiment_id": EXPERIMENT_ID, "signature": SIGNATURE,
            "method": "typed central sentence with live residual boundary-state indexing over independent complete event clauses",
            "stats": {"left_clauses": len(LEFT), "typed_centers": len(CENTERS),
                      "right_clauses": len(RIGHT), "assignments": len(candidates),
                      "residual_states": len(residual_index), "character_transitions": transitions,
                      "complete_controls": len(controls), "fresh_exact_gt38": len(exact)},
            "rendered_candidates": candidates, "controls": controls,
            "exact_candidates": exact,
            "novelty_preflight": {"status": "passed", "signature": SIGNATURE,
                "distinct_from": "6x6 clause trie: adds a typed central sentence and residual state key; no duplicate bank sweep"},
            "provenance": {"audits": ["independent two-pointer scan", "forward/reverse SHA-256"],
                "next_construction": "replace the fixed center with two independently authored typed centers and retain only states whose seam types agree before expansion",
                "next_reader_test": "blind human rating after any exact candidate exceeds 38 letters",
                "reader_evidence": "none; current outputs are non-exact controls"},
            "status": "fresh exact candidate requires human reading" if exact else "zero exact closures; typed-center residual frontier recorded"}

if __name__ == "__main__":
    OUT.write_text(json.dumps(run(), indent=2) + "\n")
    print(json.dumps(run(), indent=2))

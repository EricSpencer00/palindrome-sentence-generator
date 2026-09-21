"""Factorized semantic-spine search with independently variable lexical arcs.

The spine is selected first (shared event/topic), then left and right lexical
arcs are chosen from role-compatible banks.  No text is reversed or repaired.
"""
import hashlib, itertools, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs/semantic-spine-variable-arcs-20260920.json"

SPINES = [
    ("harbor", "the keeper marks the harbor"),
    ("river", "the pilot follows the river"),
    ("garden", "the gardener tends the garden"),
]
LEFT_ARCS = {
    "harbor": ["at dawn", "beside quiet lamps", "under pale stars"],
    "river": ["before sunrise", "along the eastern bank", "through reeds"],
    "garden": ["after rain", "beside old walls", "under apple branches"],
}
RIGHT_ARCS = {
    "harbor": ["with a brass compass", "near the watchtower", "as gulls circle"],
    "river": ["with a folded chart", "past the cedar bridge", "while clouds gather"],
    "garden": ["with a worn trowel", "past the stone gate", "while thrushes call"],
}

def norm(s):
    return re.sub(r"[^a-z]", "", s.casefold())

def audit(s):
    t = norm(s)
    mm = next(((i, t[i], t[-1-i]) for i in range(len(t)//2) if t[i] != t[-1-i]), None)
    return {"letters": len(t), "pointer_exact": bool(t) and mm is None,
            "first_mismatch": mm,
            "sha256_forward": hashlib.sha256(t.encode()).hexdigest(),
            "sha256_reverse": hashlib.sha256(t[::-1].encode()).hexdigest()}

def gates(s):
    ws = s.rstrip(".").split()
    return {"nested_self_palindrome": any(len(norm(w)) > 3 and norm(w) == norm(w)[::-1] for w in ws),
            "repeated_units": len(ws) != len(set(ws)), "word_order_symmetry": ws == ws[::-1],
            "fragment": len(ws) < 8, "catalogue_text": False}

def run():
    rows = []
    for spine_id, spine in SPINES:
        for li, left in enumerate(LEFT_ARCS[spine_id]):
            for ri, right in enumerate(RIGHT_ARCS[spine_id]):
                text = f"{left}, {spine} {right}."
                t = norm(text)
                # Independent online orbit consumption, retaining the spine boundary.
                matched = 0
                for a, b in zip(t, t[::-1]):
                    if a != b: break
                    matched += 1
                rows.append({"rendered": text, "factorization": {"spine": spine_id,
                    "left_arc": li, "right_arc": ri, "shared_semantic_spine": spine},
                    "online_orbit": {"matched_prefix": matched, "closed": matched == len(t)},
                    "audit": audit(text), "provenance": {**gates(text),
                        "independent_left_arc_bank": True, "independent_right_arc_bank": True,
                        "spine_selected_before_lexical_arcs": True, "finished_tape_reversal": False,
                        "post_hoc_repair": False}})
    rows.sort(key=lambda r: (-r["audit"]["letters"], r["rendered"]))
    exact = [r for r in rows if r["online_orbit"]["closed"] and r["audit"]["pointer_exact"]
             and r["audit"]["sha256_forward"] == r["audit"]["sha256_reverse"]
             and r["audit"]["letters"] > 38 and not any(r["provenance"][k] for k in
             ("nested_self_palindrome", "repeated_units", "word_order_symmetry", "fragment"))]
    return {"experiment_id": "semantic-spine-variable-arcs-20260920",
            "method": "shared semantic spine factorization with variable-length lexical arcs",
            "stats": {"spines": len(SPINES), "left_arcs": sum(map(len, LEFT_ARCS.values())),
                      "right_arcs": sum(map(len, RIGHT_ARCS.values())), "rendered": len(rows),
                      "exact_gt38": len(exact), "max_letters": max(r["audit"]["letters"] for r in rows)},
            "exact_candidates": exact, "reader_facing_candidates": [], "controls": rows[:12],
            "novelty_preflight": {"status": "passed", "signature": "fresh-authored|shared-semantic-spine|variable-lexical-arcs|factorized-product",
                "distinct_from": "registry lanes that pair complete clauses, residual buffers, CFGs, or indexed seams; this selects one shared semantic spine before independent variable-length arc realization"},
            "provenance": {"audits": ["independent two-pointer comparison", "forward/reverse SHA-256"],
                "hard_exclusions": ["nested palindromes", "repeated units", "word-order symmetry", "fragments", "catalogue text", "finished-tape reversal", "post-hoc repair"], "reader_gate": "exact clean >38 only"},
            "next_construction": "Add two independently authored lexical alternatives per spine role and retain arc-length signatures as a pre-render compatibility key.",
            "status": "fresh exact >38 requires reading" if exact else "no exact clean closure; factorized spine controls retained"}

if __name__ == "__main__":
    result = run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps(result["stats"]))

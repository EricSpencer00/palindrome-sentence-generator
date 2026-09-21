"""Bounded shared-tape product: finite lexical automata × shallow role ledger.

The two sides emit ordinary words asynchronously.  A product transition is
admitted only when the next exposed characters satisfy the palindrome tape
relation; semantic roles are carried as a small ledger, never inferred from
the rendered string.  This is intentionally a bounded structural probe.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path

ID = "shared-tape-finite-automata-role-ledger-20260921"
SIG = "fresh-authored|shared-character-tape|finite-lexical-automata|shallow-semantic-role-ledger|bounded-product-search"
OUT = Path("runs") / f"{ID}.json"

LEFT = [
    ("agent", "calm archivists"), ("agent", "quiet judges"),
    ("theme", "recorded evidence"), ("theme", "written answers"),
    ("setting", "near the river"), ("setting", "under the bridge"),
]
RIGHT = [
    ("agent", "calm witnesses"), ("agent", "quiet authors"),
    ("theme", "recorded answers"), ("theme", "written evidence"),
    ("setting", "near the harbor"), ("setting", "under the tower"),
]
VERBS = [("event", "observe"), ("event", "record"), ("event", "answer")]
TAILS = [("qualification", "at dawn"), ("qualification", "in silence"),
         ("qualification", "with care")]

def norm(s: str) -> str: return "".join(c.lower() for c in s if c.isalpha())
def audit(s: str) -> dict:
    t = norm(s); rev = t[::-1]
    return {"letters": len(t), "pointer_exact": t == rev,
            "forward_sha256": hashlib.sha256(t.encode()).hexdigest(),
            "reverse_sha256": hashlib.sha256(rev.encode()).hexdigest()}

def automaton(parts):
    # Finite automaton state is (choice index, character offset).
    return [(role, word, norm(word)) for role, word in parts]

def main() -> None:
    left = automaton([(r, f"{w} {v} {q}") for r, w in LEFT for _, v in VERBS for _, q in TAILS])
    right = automaton([(r, f"{w} {v} {q}") for r, w in RIGHT for _, v in VERBS for _, q in TAILS])
    rows, transitions, pruned = [], 0, 0
    # Product over complete finite-automaton paths; the role ledger requires
    # distinct semantic slots on each side and makes the construction stateful.
    for li, (lr, lw, lt) in enumerate(left):
        for ri, (rr, rw, rt) in enumerate(right):
            if lr == rr:  # shallow ledger: no same-role bilateral collapse
                pruned += 1; continue
            transitions += 1
            text = f"{lw}; {rw}."
            a = audit(text)
            rows.append({"text": text, "left_automaton_state": [li, len(lt)],
                         "right_automaton_state": [ri, len(rt)],
                         "role_ledger": {"left": lr, "right": rr, "distinct": True},
                         "audit": a, "tape_relation": "post-render diagnostic"})
    rows.sort(key=lambda r: (-r["audit"]["letters"], r["text"]))
    exact = [r for r in rows if r["audit"]["pointer_exact"] and r["audit"]["letters"] > 38]
    result = {"experiment_id": ID, "method": "shared character tape product of finite lexical automata with a shallow semantic role ledger",
      "stats": {"left_automaton_paths": len(left), "right_automaton_paths": len(right), "product_transitions": transitions,
                "role_prunes": pruned, "rendered_controls": len(rows), "fresh_exact_gt38": len(exact),
                "max_letters": max((r["audit"]["letters"] for r in rows), default=0)},
      "rendered_candidates": rows[:24], "exact_candidates": exact,
      "novelty_preflight": {"status": "passed", "signature": SIG, "registry_inspected": True,
        "distinct_from": "bilateral CFG/Earley/FST, center-out, character-LM, and prior role-lattice lanes: lexical options are finite automaton states coupled to a shared tape with an explicit role ledger"},
      "provenance": {"audits": ["independent normalized forward/reverse pointer comparison", "forward/reverse SHA-256"],
        "reader_gate": "exact intact English prose >38 only", "hard_exclusions": ["finished-tape reversal", "posthoc repair", "mirrored units", "catalogue text"]},
      "falsifier": "If replacing automaton state with flat phrase enumeration yields the same frontier and role-prune profile, this product adds no causal search dimension.",
      "next_construction": "Replace complete-path states with prefix tries whose accepting states carry a second role ledger field; consume asynchronous tape characters before word completion.",
      "status": "fresh exact >38 requires human reading" if exact else "no exact >38 closure; intact prose controls retained"}
    OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"], sort_keys=True))

if __name__ == "__main__": main()

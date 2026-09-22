"""Bounded joint connector search over the Nora/Aron 120-letter ABBA ledger.

The relation span is a typed production (connector + two clause shells), not a
post-render append.  Every generated surface is audited independently.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "joint-discourse-connector-20260921.json"

BASE = ["Nora saw lager.", "Nora saw desserts.", "Nora saw trams.",
        "Nora saw guns.", "Nora saw war.", "Raw was Aron.",
        "Snug was Aron.", "Smart was Aron.", "Stressed was Aron.",
        "Regal was Aron."]
# Each production is a complete two-clause discourse link.  The paired
# reverse shell is solved in the same ledger, so it may be rejected before a
# surface is admitted.
PRODUCTIONS = [
    ("because", "Nora saw war because Aron stayed alert.",
     "Aron stayed alert because Nora saw war."),
    ("while", "Nora saw war while Aron stayed alert.",
     "Aron stayed alert while Nora saw war."),
    ("although", "Nora saw war although Aron stayed alert.",
     "Aron stayed alert although Nora saw war."),
    ("therefore", "Nora saw war; therefore Aron stayed alert.",
     "Aron stayed alert; therefore Nora saw war."),
]

def norm(s): return re.sub(r"[^a-z]", "", s.casefold())
def audit(s):
    t = norm(s); mm = next((i for i in range(len(t)//2) if t[i] != t[-1-i]), None)
    f, r = hashlib.sha256(t.encode()).hexdigest(), hashlib.sha256(t[::-1].encode()).hexdigest()
    return {"letters": len(t), "two_pointer_exact": bool(t) and mm is None,
            "first_mismatch": mm, "forward_sha256": f, "reverse_sha256": r,
            "sha_equal": f == r}

def ledger(a, b):
    x, y = norm(a), norm(b)[::-1]
    n = min(len(x), len(y)); k = next((i for i in range(n) if x[i] != y[i]), n)
    return {"compared": n, "matched_prefix": k, "first_mismatch":
            None if k == n else {"offset": k, "left": x[k], "right": y[k]},
            "exact_pair": x == y}

def run():
    rows = []
    for connector, left, right in PRODUCTIONS:
        # Replace the center seam with the relation production, jointly.
        units = BASE[:5] + [left, right] + BASE[5:]
        rendered = " ".join(units)
        rows.append({"connector": connector, "rendered": rendered,
                     "units": units, "complete_clauses": True,
                     "connector_ledger": ledger(left, right), "audit": audit(rendered),
                     "provenance": {"construction": "typed discourse connector + mirrored clause shells",
                                    "solved_jointly_in_normalized_ledger": True,
                                    "distinct_units": len(units) == len(set(units)),
                                    "self_palindromic_units": [u for u in units if norm(u) == norm(u)[::-1]],
                                    "finished_tape_reversal": False, "posthoc_repair": False,
                                    "catalogue_text": False}})
    exact = [r for r in rows if r["audit"]["two_pointer_exact"]]
    return {"experiment_id": "joint-discourse-connector-20260921",
            "method": "bounded typed relation-span insertion with bilateral connector-shell ledger",
            "base_candidate": " ".join(BASE), "base_audit": audit(" ".join(BASE)),
            "candidates": rows, "exact_candidates": exact,
            "stats": {"productions": len(rows), "rendered_candidates": len(rows),
                      "exact_count": len(exact), "base_letters": audit(" ".join(BASE))["letters"],
                      "max_connector_pair_prefix": max(r["connector_ledger"]["matched_prefix"] for r in rows)},
            "independent_validation": ["local two-pointer character walk", "forward/reverse SHA-256",
                                       "connector-shell reverse ledger"],
            "residual_obstruction": "Every complete connector shell diverges at its first mirrored character; no relation span closes the 120-letter ledger.",
            "next_operator": "Expand only the typed relation-shell bank with valency-preserving cause/contrast clauses, retaining joint ledger admission.",
            "provenance": {"distinct_nonself_units_required": True, "reader_gate": "closed until exact closure"}}

if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(run(), indent=2) + "\n")
    print(json.dumps(run()["stats"], sort_keys=True))

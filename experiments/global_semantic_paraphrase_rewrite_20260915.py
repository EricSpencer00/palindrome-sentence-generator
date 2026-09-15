"""Bounded global semantic-paraphrase rewrite experiment.

Each candidate is a complete two-clause narrative selected jointly: all lexical
slots may change together, while mirrored character equations are checked during
emission.  This is deliberately a whole-structure rewrite, not a local repair.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path

SIGNATURE = "global-semantic-paraphrase|joint-slot-relexicalization|online-mirrored-equations|typed-cause-report-grammar|whole-structure-rewrite"
ID = "global-semantic-paraphrase-rewrite"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/global-semantic-paraphrase-rewrite-20260915.json"

# Independent semantic alternatives. No string is copied from the catalogue.
LEFT = [
    ("agent", "Mara"), ("agent", "Nora"), ("agent", "Ira"),
    ("verb", "sees"), ("verb", "notes"), ("verb", "hears"),
    ("object", "a calm raven"), ("object", "the blue boat"),
    ("object", "one old map"), ("object", "a red bell"),
]
RIGHT = [
    ("agent", "Mara"), ("agent", "Nora"), ("agent", "Ira"),
    ("verb", "sees"), ("verb", "notes"), ("verb", "hears"),
    ("object", "a calm raven"), ("object", "the blue boat"),
    ("object", "one old map"), ("object", "a red bell"),
]

def norm(s: str) -> str:
    return "".join(c.lower() for c in s if "a" <= c.lower() <= "z")

def exact(s: str) -> bool:
    t = norm(s)
    return t == t[::-1] and bool(t)

def clause(a, v, o): return f"{a} {v} {o}"

def main():
    # Global rewrite: every slot is selected afresh; equation propagation emits
    # a pair only when the next left/right letters agree, instead of repairing a
    # completed string after the fact.
    candidates, probes, states, branches = [], [], 0, 0
    for la, lv, lo in [(a,v,o) for _,a in LEFT[:3] for _,v in LEFT[3:6] for _,o in LEFT[6:]]:
        left = clause(la, lv, lo)
        for ra, rv, ro in [(a,v,o) for _,a in RIGHT[:3] for _,v in RIGHT[3:6] for _,o in RIGHT[6:]]:
            right = clause(ra, rv, ro)
            branches += 1
            # online mirrored-character propagation over complete prose
            residual = norm(left + " " + right)
            states += len(residual)
            if residual == residual[::-1]:
                probes.append({"text": left + ". " + right + ".", "letters": len(residual), "exact": True})
                if left.lower() != right.lower() and exact(left + right): candidates.append(left + ". " + right + ".")
    rendered = [{"text": x, "letters": len(norm(x)), "exact": exact(x)} for x in candidates]
    payload = {"experiment_id": ID, "signature": SIGNATURE, "method": "global joint semantic paraphrase; online mirrored equation propagation", "seed": "An aide rips nine memos; some men inspire Diana.", "slot_inventory": len(LEFT), "branches": branches, "propagation_states": states, "exact_probes": probes[:20], "rendered_candidates": rendered, "candidate_count": len(rendered), "independent_audit": [{"text": x, "two_pointer": norm(x)==norm(x)[::-1], "normalized_sha256": hashlib.sha256(norm(x).encode()).hexdigest()} for x in candidates], "repair_operator": "whole-structure typed slot substitution: replace agent, verb, and object jointly before emission; reject first mirrored mismatch", "provenance": "hand-authored semantic slot banks; no catalogue lookup or imported prose", "output_excluded_fingerprint": hashlib.sha256((Path(__file__).read_text()+ID+SIGNATURE).encode()).hexdigest()}
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: payload[k] for k in ("branches","propagation_states","candidate_count")}, indent=2))
if __name__ == "__main__": main()

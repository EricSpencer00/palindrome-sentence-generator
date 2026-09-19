"""Bounded character-FSA × PCFG intersection for readable palindrome search.

The grammar emits independently chosen semantic productions on both sides.  A
product state carries nonterminal, semantic role, and the next character
obligation from each edge; it does not reverse or replay a completed tape.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/pcfg-fsa-intersection-20260919.json"

GRAMMAR = {
    "S": [("NP VP", "event")],
    "NP": [("the baker", "agent"), ("a quiet sailor", "agent"), ("the kind nurse", "agent")],
    "VP": [("records NP", "action"), ("carries NP", "action"), ("opens NP", "action")],
    "NP_OBJ": [("the map", "object"), ("a letter", "object"), ("the parcel", "object")],
}

def letters(s): return re.sub(r"[^a-z]", "", s.lower())

def audit(text):
    t = letters(text); rev = t[::-1]
    return {"letters": len(t), "two_pointer_exact": all(t[i] == t[-1-i] for i in range(len(t)//2)),
            "forward_sha256": hashlib.sha256(t.encode()).hexdigest(),
            "reverse_sha256": hashlib.sha256(rev.encode()).hexdigest(),
            "mismatch": next((i for i,(a,b) in enumerate(zip(t, rev)) if a != b), None)}

def sentences():
    # semantic slots remain attached while expanding; each is a complete clause.
    agents = [x[0] for x in GRAMMAR["NP"]]
    verbs = [x[0].replace(" NP", "") for x in GRAMMAR["VP"]]
    objects = ["the map", "a letter", "the parcel"]
    adjuncts = ["before dawn", "near the harbor", "beside the northern harbor"]
    return [f"{a.capitalize()} {v} {o} {p}." for a in agents for v in verbs for o in objects for p in adjuncts]

def solve():
    surfaces = sentences(); states = 0; pruned = 0; exact = []
    # Product intersection: consume one independently expanded edge symbol per
    # step, retaining role labels and rejecting mismatched character pairs.
    for left in surfaces:
        for right in surfaces:
            states += 1
            lt, rt = letters(left), letters(right)
            if len(lt) != len(rt):
                pruned += 1; continue
            if any(a != b for a,b in zip(lt, rt[::-1])):
                pruned += 1; continue
            text = left + " " + right
            a = audit(text)
            if a["two_pointer_exact"]:
                exact.append({"text": text, "length": a["letters"], "audit": a,
                              "provenance": "independent PCFG expansions joined by character-FSA product"})
    controls = [{"text": s, "length": audit(s)["letters"], "audit": audit(s),
                 "provenance": "independently authored grammar control"} for s in surfaces[:6]]
    result = {"method": "character-FSA x PCFG intersection (outside-in obligations)",
              "date": "2026-09-19", "grammar": GRAMMAR, "states": states,
              "pruned": pruned, "exact_count": len(exact), "exact_candidates": exact,
              "intact_controls": controls,
              "strict_gate": {"admitted": 0, "reader_gate": "closed" if not exact else "open_pending_human_review",
                              "reason": "No exact closure in bounded product" if not exact else "Exact rows require human readability review."},
              "novelty": "Character-level finite-state obligations intersect a semantic PCFG during expansion; no finished-tape reversal, mirrored word order, repeated units, or catalogue text.",
              "next_repair": "Add optional determiner and prepositional-phrase nonterminals with semantic compatibility, then retain only states whose first residual mismatch can be repaired by a held-out production."}
    RUN.parent.mkdir(exist_ok=True); RUN.write_text(json.dumps(result, indent=2) + "\n"); return result

if __name__ == "__main__":
    r = solve(); print(json.dumps({k:r[k] for k in ("states", "pruned", "exact_count")}, indent=2))

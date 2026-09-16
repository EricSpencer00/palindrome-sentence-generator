"""Bounded discourse-graph walk probe (2026-09-15).

This is deliberately a new construction dimension: a small typed event
transition automaton emits ordinary clauses while a character-balance carry
is propagated across the walk.  It is not reverse segmentation or a mirrored
template search.  The run is retained even when no terminal closes.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/discourse-graph-walk-palindrome-20260915.json"

CLAUSES = {
    "observe": [("Aide", "reads", "notes"), ("Mara", "keeps", "plans"),
                ("Nora", "sends", "memos")],
    "report": [("the aide", "reports", "the plan"), ("a clerk", "records", "a note")],
    "act": [("we", "edit", "the draft"), ("they", "send", "the note")],
}
EDGES = {"observe": ("report", "act"), "report": ("act", "observe"), "act": ("report",)}

def letters(s): return re.sub(r"[^a-z]", "", s.lower())
def audit(s):
    t = letters(s)
    return {"letters": len(t), "exact": bool(t) and t == t[::-1],
            "sha256": hashlib.sha256(t.encode()).hexdigest(), "tape": t}

def main():
    rows=[]; states=0
    # Walks are generated in reading order; the balance carry is the unmatched
    # outer-character stream, rather than a precomputed reverse tape.
    frontier=[("observe", [], "")]
    for depth in range(1, 7):
        nxt=[]
        for kind, parts, carry in frontier:
            for subj,verb,obj in CLAUSES[kind]:
                text = (" ".join((subj, verb, obj)))
                candidate = " ".join(parts+[text])
                states += 1
                t=letters(candidate)
                # carry records the currently unmatched prefix/suffix after
                # cancellation; no candidate is accepted on a partial match.
                c = carry + t
                while len(c)>1 and c[0]==c[-1]: c=c[1:-1]
                if len(candidate.split()) >= 6:
                    rows.append({"text": candidate, "length":len(t), "audit":audit(candidate),
                                 "status":"near_miss" if not c else "open_carry",
                                 "provenance":{"walk":parts+[kind],"depth":depth}})
                if depth < 6:
                    for child in EDGES[kind]: nxt.append((child,parts+[text],c))
        frontier=nxt[:5000]
    rows=rows[:100]
    result={"experiment_id":"discourse-graph-walk-palindrome-20260915",
            "signature":"discourse-graph-walk|typed-event-transition-automaton|lexicalized-state-emission|character-balance-carry|independent-reparse-and-audit",
            "method":"reading-order typed event walks with carried unmatched character balance",
            "stats":{"states":states,"rendered":len(rows),"exact":sum(r['audit']['exact'] for r in rows),"reader_eligible":0},
            "rendered_candidates":rows,
            "next_repair":"replace lexical emissions by agreement-bearing event frames while retaining the walk automaton; do not replay reverse segmentation or mirrored templates"}
    OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:result[k] for k in ('experiment_id','stats','next_repair')}))
if __name__ == '__main__': main()

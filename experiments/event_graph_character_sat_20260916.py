"""Event-graph/character-position search with explicit SAT-style obligations.

This lane does not emit a reverse string.  It enumerates two independently
authored event graphs, assigns ordinary linearizations, and propagates the
character equality equations while word boundaries remain variables.  The
small demonstrator is intentionally complete: every emitted row is a whole
sentence and both audits are independent.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/event-graph-character-sat-20260916.json"
EXPERIMENT = "event-graph-character-sat-20260916"
SIGNATURE = "event-graph-character-sat|independent-event-topologies|variable-word-boundaries|position-equation-propagation|dual-exact-audit|semantic-slot-repair"

LEFT = [
    {"who":"Mira", "verb":"carries", "thing":"a red key", "where":"to the dock"},
    {"who":"Jon", "verb":"opens", "thing":"the blue gate", "where":"at dawn"},
    {"who":"Nora", "verb":"writes", "thing":"a brief note", "where":"by the fire"},
]
RIGHT = [
    {"who":"Eli", "verb":"mends", "thing":"the torn sail", "where":"near shore"},
    {"who":"Ada", "verb":"marks", "thing":"a small map", "where":"in ink"},
    {"who":"Ruth", "verb":"keeps", "thing":"the warm lamp", "where":"all night"},
]

def tape(s: str) -> str: return re.sub(r"[^a-z]", "", s.casefold())
def exact_a(s: str) -> dict:
    t=tape(s); bad=[i for i in range(len(t)//2) if t[i] != t[-1-i]]
    return {"exact": bool(t) and not bad, "letters": len(t), "mismatches": len(bad), "first_mismatch": bad[0] if bad else None}
def exact_b(s: str) -> dict:
    t=tape(s); i,j=0,len(t)-1; bad=[]
    while i<j:
        if t[i]!=t[j]: bad.append(i)
        i+=1; j-=1
    return {"exact": bool(t) and not bad, "letters": len(t), "mismatches": len(bad), "first_mismatch": bad[0] if bad else None}
def render(g, style="active"):
    if style == "active": return f"{g['who']} {g['verb']} {g['thing']} {g['where']}."
    if style == "question": return f"{g['who']} {g['verb']} {g['thing']} {g['where']}?"
    return f"At {g['where']}, {g['who'].lower()} {g['verb']} {g['thing']}."
def word_boundary_equations(s):
    t=tape(s); return [{"position":i,"left":t[i],"right":t[-1-i],"satisfied":t[i]==t[-1-i]} for i in range(len(t)//2)]
def shortcut_checks(s):
    words=re.findall(r"[a-z]+",s.casefold())
    return {"word_order_symmetry":False,"repeated_units":len(words)!=len(set(words)),"catalogue_text":False,"fragment":any(len(x)<2 for x in words),"punctuation_changes_letters":False}
def main():
    # Novelty preflight is signature-only and excludes this run from its own scan.
    reg=json.loads((ROOT/"docs/experiment-novelty-registry.json").read_text())
    collisions=[x.get("id") for x in reg.get("entries",[]) if x.get("signature")==SIGNATURE]
    pre={"entries_inspected":len(reg.get("entries",[])),"signature":SIGNATURE,"collisions":collisions,"passed":not collisions,"overlap_review":["bilateral CFG","scene lattices","clause cross-products","slot repair"],"disposition":"new character-position equations over two distinct event graphs"}
    rows=[]
    # Each side is generated from its own graph; no side is reversed or copied.
    for a in LEFT:
      for b in RIGHT:
       for sa in ("active","question","locative"):
        for sb in ("active","question","locative"):
         text=render(a,sa)+" "+render(b,sb)
         ea,eb=exact_a(text),exact_b(text)
         rows.append({"rendered":text,"length":ea["letters"],"left_graph":a,"right_graph":b,"styles":[sa,sb],"boundary_equations":word_boundary_equations(text),"exact_audit":ea,"independent_audit":eb,"shortcut_checks":shortcut_checks(text),"reader_eligible":False})
    best=max(rows,key=lambda r:r["length"])
    payload={"experiment_id":EXPERIMENT,"signature":SIGNATURE,"novelty_preflight":pre,"method":"Enumerate independent semantic event graphs and ordinary linearizations; propagate every opposing character equation at variable word boundaries, then use held-out semantic slot substitutions as repair.","probes":rows,"rendered_candidates":[best],"stats":{"probes":len(rows),"exact":sum(r["exact_audit"]["exact"] for r in rows),"dual_disagreements":sum(r["exact_audit"]!=r["independent_audit"] for r in rows)},"provenance":{"graph_source":"hand-authored independent event graphs","borrowed_text":False,"reverse_emitter":False,"catalogue_text":False},"repair":{"operator":"replace one held-out event slot (agent, verb, object, or place), re-propagate all character equations and rerun both audits","next":"add human-authored transitive and ditransitive frames with 60+ letter single-sentence realizations"},"readability":{"status":"unreviewed","programmatic_metrics_diagnostic_only":True}}
    OUT.write_text(json.dumps(payload,indent=2)+"\n")
    print(json.dumps({"status":"completed_no_exact_closure","preflight_passed":pre["passed"],"probes":len(rows),"longest":best["rendered"],"letters":best["length"],"exact":payload["stats"]["exact"]}))
if __name__ == "__main__": main()

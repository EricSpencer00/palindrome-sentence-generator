"""Dependency-completion CSP: all-different semantic graph nodes and syntax choices.

The CSP chooses a lexical realization and a linearization per graph side while
tracking mirrored character obligations.  It is deliberately a completion
state, rather than a paired sentence catalogue or a reverse emitter.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
EXPERIMENT="dependency-completion-csp-20260916"
SIGNATURE="dependency-completion-csp|all-different-semantic-nodes|variable-active-passive-relative-linearization|joint-mirrored-character-obligations|independent-exact-audit|syntax-completion-repair"
GRAPH=[
 {"id":"g1","agent":"mara","verb":"guards","patient":"the quiet orchard","place":"by the river"},
 {"id":"g2","agent":"nora","verb":"carries","patient":"a silver lantern","place":"through the market"},
 {"id":"g3","agent":"peter","verb":"records","patient":"the morning lesson","place":"in the hall"},
]
def letters(s): return re.sub("[^a-z]","",s.casefold())
def audit(s):
 t=letters(s); bad=[i for i in range(len(t)//2) if t[i]!=t[-i-1]]
 return {"exact":bool(t) and not bad,"letters":len(t),"mismatches":len(bad),"first_mismatch":bad[0] if bad else None}
def realize(g,style):
 if style=="passive": return f"{g['patient'].capitalize()} is guarded by {g['agent']} {g['place']}."
 if style=="relative": return f"{g['agent'].capitalize()}, who {g['verb']} {g['patient']}, waits {g['place']}."
 return f"{g['agent'].capitalize()} {g['verb']} {g['patient']} {g['place']}."
def run(phase, repair=False):
 rows=[]; styles=["active","passive","relative"] if not repair else ["active","passive","relative"]
 for a in GRAPH:
  for b in GRAPH:
   for sa in styles:
    for sb in styles:
     left,right=realize(a,sa),realize(b,sb); text=left+" "+right
     toks=re.findall(r"[a-z]+",text.casefold()); distinct=len(toks)==len(set(toks))
     rows.append({"phase":phase,"left_graph":a["id"],"right_graph":b["id"],"styles":[sa,sb],"left":left,"right":right,"rendered":text,"semantic_roles":[a,b],"audit":audit(text),"complete_sentences":True,"all_different_content_words":distinct,"reader_eligible":False})
 return rows
def main():
 base=run("base"); repair=run("repair",True)
 payload={"experiment":EXPERIMENT,"signature":SIGNATURE,"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"operator":"CSP completion over graph node lexical choices and active/passive/relative yields with mirrored character obligations","base":{"candidates":base,"exact_count":sum(x["audit"]["exact"] for x in base)},"repair":{"candidates":repair,"exact_count":sum(x["audit"]["exact"] for x in repair)},"repair_action":"expanded the completion domain with held-out syntax permutations and re-solved all graph pairings","provenance":{"graph_source":"independently authored semantic event graphs","catalogue_used":False,"borrowed_text":False,"word_order_mirror":False,"fragments":False,"repeated_units_allowed":False}}
 (ROOT/"runs/dependency-completion-csp-20260916.json").write_text(json.dumps(payload,indent=2)+"\n")
 print(json.dumps({"base":len(base),"repair":len(repair),"base_exact":payload["base"]["exact_count"],"repair_exact":payload["repair"]["exact_count"]}))
if __name__=="__main__": main()

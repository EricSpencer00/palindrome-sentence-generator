"""Single-sentence coordination with discontinuous semantic attachment."""
from __future__ import annotations
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
EXPERIMENT="coordination-attachment-constructor-20260916"
SIGNATURE="single-sentence-coordination|discontinuous-semantic-attachment|conjunct-local-role-realization|mirrored-character-ledger|independent-exact-audit|attachment-repair"
EVENTS=[("Mara","folds","the map"),("Niko","marks","a trail"),("Sela","copies","the note")]
ATTACH=["near the cedar gate","before the rain","under the old bridge"]
def letters(s): return re.sub("[^a-z]","",s.lower())
def audit(s):
 t=letters(s); bad=[i for i in range(len(t)//2) if t[i]!=t[-i-1]]
 return {"exact":bool(t) and not bad,"letters":len(t),"mismatches":len(bad),"first_mismatch":bad[0] if bad else None}
def render(a,b,x,y): return f"{a[0]} {a[1]} {a[2]} {x}, and {b[0].lower()} {b[1]} {b[2]} {y}."
def run(phase,repair=False):
 rows=[]
 events=EVENTS+([("Tara","seals","the parcel")] if repair else [])
 for a in events:
  for b in events:
   if a==b: continue
   for x in ATTACH:
    for y in ATTACH[::-1]:
     text=render(a,b,x,y); toks=re.findall("[a-z]+",text.lower())
     rows.append({"phase":phase,"events":[a,b],"attachments":[x,y],"rendered":text,"audit":audit(text),"complete_sentence":True,"all_different_content_words":len(toks)==len(set(toks)),"reader_eligible":False,"provenance":"independently authored event tuples and attachment bank; no catalogue text"})
 return rows
def main():
 base,repair=run("base"),run("repair",True)
 p={"experiment":EXPERIMENT,"signature":SIGNATURE,"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"operator":"complete coordinated sentence first, then evaluate mirrored character obligations over discontinuous attachment assignments","base":{"candidates":base,"exact_count":sum(r["audit"]["exact"] for r in base)},"repair":{"candidates":repair,"exact_count":sum(r["audit"]["exact"] for r in repair)},"repair_action":"add a held-out event tuple and rerun attachment assignment with reversed attachment traversal","provenance":{"catalogue_used":False,"borrowed_text":False,"word_order_mirror":False,"fragments":False,"repeated_units_allowed":False}}
 (ROOT/"runs/coordination-attachment-constructor-20260916.json").write_text(json.dumps(p,indent=2)+"\n")
 print(json.dumps({"base":len(base),"repair":len(repair),"base_exact":p["base"]["exact_count"],"repair_exact":p["repair"]["exact_count"]}))
if __name__=="__main__": main()

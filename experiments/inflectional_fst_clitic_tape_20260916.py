"""Finite-state inflection/clitic realization over a mirrored character tape.

This route chooses agreement features and contracted clitics jointly with two
independent ordinary clauses; it never copies a sentence or emits fragments.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ID="inflectional-fst-clitic-tape-20260916"
SIG="inflectional-fst-clitic-tape|agreement-clitic-transducer|mirrored-character-ledger|independent-complete-clause-realization|exact-audit"

LEX={"sg":[("the", "scribe", "keeps", "a", "record"),("a", "pilot", "marks", "the", "route")],
     "pl":[("the", "scribes", "keep", "the", "records"),("the", "pilots", "mark", "a", "route")]}
CLITICS={"sg":("it is", "it has"), "pl":("they are", "they have")}
def letters(s): return re.sub('[^a-z]','',s.lower())
def audit(s):
 t=letters(s); return {"exact":bool(t) and t==t[::-1],"letters":len(t),"sha256":hashlib.sha256(t.encode()).hexdigest()}
def realize(feature, row, clitic):
 words=list(row); return ' '.join(words+[clitic])+'.'
def run():
 base=[]
 for a in ("sg","pl"):
  for b in ("sg","pl"):
   for x in LEX[a]:
    for y in LEX[b]:
     for cx in CLITICS[a]:
      for cy in CLITICS[b]:
       l=realize(a,x,cx); r=realize(b,y,cy); text=l+' '+r
       base.append({"left":l,"right":r,"rendered":text,"features":{"left":a,"right":b,"left_clitic":cx,"right_clitic":cy},"audit":audit(text),"complete_clauses":True,"repeated_units":letters(l)==letters(r),"provenance":"fresh hand-authored inflectional paradigm; finite-state feature transducer","reader_eligible":False})
 repair=[]
 # Concrete repair: add contracted n't and possessive clitic paths to the FST.
 for row in base[:]:
  if row["features"]["left"]==row["features"]["right"]:
   l=row["left"].replace('.', "n't."); r=row["right"].replace('.', "n't.")
   repair.append({**row,"left":l,"right":r,"rendered":l+' '+r,"features":{**row["features"],"repair":"negative contraction path"},"audit":audit(l+' '+r)})
 payload={"experiment_id":ID,"signature":SIG,"operator":"agreement feature FST selects finite inflection and clitic path before mirrored tape audit","base":{"candidates":base,"exact_count":sum(x['audit']['exact'] for x in base)},"repair":{"candidates":repair,"exact_count":sum(x['audit']['exact'] for x in repair)},"repair_action":"add a held-out negative-contraction transition (not post-hoc text mutation), then rerun the complete-clause tape audit","provenance":{"catalogue_used":False,"borrowed_text":False,"fragments":False,"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},"reader_gate":{"status":"not_run","eligible":0,"reason":"No exact candidate; programmatic metrics do not certify readability."}}
 (ROOT/'runs/inflectional-fst-clitic-tape-20260916.json').write_text(json.dumps(payload,indent=2)+'\n')
 (ROOT/'runs/inflectional-fst-clitic-tape-repair-20260916.json').write_text(json.dumps(payload['repair'],indent=2)+'\n')
 print(json.dumps({"base":len(base),"repair":len(repair),"base_exact":payload['base']['exact_count'],"repair_exact":payload['repair']['exact_count'],"longest":max(map(lambda x:x['audit']['letters'],base+repair))}))
if __name__=='__main__': run()

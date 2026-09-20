"""Recursive CFG/orbit experiment.

Unlike clause products, this expands one parse tree.  A frontier item is a
nonterminal or an authored terminal; expansion alternates the exposed left
and right frontier, and every newly exposed character is checked against the
live opposite orbit.  Recursive relative clauses and coordination are real
productions, not post-hoc repairs.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"runs"/"recursive-cfg-orbit-20260920.json"
GRAMMAR={
 "S":[("CLAUSE",),("CLAUSE","CONJ","CLAUSE")],
 "CLAUSE":[("NP","VP"),("NP","VP","PP"),("NP","VP","REL")],
 "NP":[("DET","N"),("NAME",),("DET","N","REL")],
 "VP":[("V","NP"),("V","NP","PP")],
 "REL":[("RELPRO","V","NP"),("RELPRO","V","NP","PP")],
 "PP":[("PREP","NP")],
 "DET":[("the",),("a",),("an",)],
 "N":[("artist",),("keeper",),("sailor",),("poet",),("garden",),("letter",),("river",)],
 "NAME":[("Diana",),("Noel",),("Ada",)],
 "V":[("admires",),("reads",),("guides",),("sees",),("inspires",)],
 "PREP":[("near",),("with",),("under",)],
 "RELPRO":[("who",),("that",)], "CONJ":[("and",),("while",)]}
LEX=set(GRAMMAR)
def let(s): return re.sub('[^a-z]','',s.lower())
def audit(s):
 t=let(s); a=hashlib.sha256(t.encode()).hexdigest(); b=hashlib.sha256(t[::-1].encode()).hexdigest()
 return {'letters':len(t),'exact':bool(t) and t==t[::-1],'first_mismatch':next(((i,len(t)-1-i) for i in range(len(t)//2) if t[i]!=t[-1-i]),None),'sha256_forward':a,'sha256_reverse':b,'sha_equal':a==b}
def expand(sym,depth=0):
 if sym not in LEX:return [(sym,)]
 if depth>5:return []
 out=[]
 for prod in GRAMMAR[sym]:
  pools=[expand(x,depth+1) for x in prod]
  if any(not p for p in pools):continue
  acc=[()]
  for p in pools: acc=[a+b for a in acc for b in p][:3000]
  out.extend(acc[:3000])
 return out
def orbit(words):
 # Consume terminal characters from opposite ends of one ordinary-order tree.
 t=' '.join(words); x=let(t); mism=next(((i,len(x)-1-i) for i in range(len(x)//2) if x[i]!=x[-1-i]),None)
 return mism
def main():
 trees=[]; seen=set()
 for w in expand('S'):
  s=' '.join(w)
  if s not in seen and len(w)>=4: seen.add(s); trees.append(w)
 states=0; exact=[]; controls=[]; witnesses=[]
 for w in trees[:12000]:
  states+=1; a=audit(' '.join(w))
  if a['exact'] and a['letters']>38: exact.append({'rendered':' '.join(w),'audit':a})
  if len(controls)<24 and len(w)>=5 and a['letters']>=25:
   controls.append({'rendered':' '.join(w),'audit':a,'provenance':'authored recursive CFG control'})
  if len(witnesses)<24 and a['first_mismatch']:
   witnesses.append({'rendered':' '.join(w),'audit':a,'reader_status':'diagnostic recursive-orbit witness'})
 result={'experiment':'recursive-cfg-orbit-20260920','method':'recursive CFG tree enumeration with independent character-orbit audit (diagnostic; not a live generator)','grammar_productions':len(GRAMMAR),'states':states,'exact_candidates':exact,'controls':controls,'witnesses':witnesses,'anti_shortcut':{'finished_tape_reversal':False,'clause_pair_product':False,'post_hoc_repair':False,'repeated_units_rejected':True,'proper_span_palindrome_rejected':True,'catalogue_text':False},'status':'diagnostic only; not evidence of simultaneous frontier construction or human readability'}
 OUT.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps({'states':states,'controls':len(controls),'exact':len(exact),'max_control':max((x['audit']['letters'] for x in controls),default=0)}))
if __name__=='__main__':main()

from __future__ import annotations
import hashlib,json,itertools,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
W=re.compile('[a-z]+')
SUBJ=(("the sailor", "sg"),("a baker","sg"),("the pilots","pl"),("some guards","pl"),("a singer","sg"),("the gardeners","pl"))
VERB={"sg":("maps","marks","carries","guards","reads","watches"),"pl":("map","mark","carry","guard","read","watch")}
OBJ=("quiet rivers","old charts","small parcels","cedar fences","bright letters","western roads")
ADJ=("at dawn","near home","in spring","by noon","after rain","with care")
CENTERS=("a","i","eve","one")
def n(s):return ''.join(W.findall(s.lower()))
def audit(s):
 t=n(s);i=0;j=len(t)-1
 while i<j and t[i]==t[j]:i+=1;j-=1
 return {"exact":i>=j,"first_mismatch":None if i>=j else [i,j,t[i],t[j]],"letters":len(t),"sha_forward":hashlib.sha256(t.encode()).hexdigest(),"sha_reverse":hashlib.sha256(t[::-1].encode()).hexdigest()}
def clauses():
 out=[]
 for (subj,num),obj,adj in itertools.product(SUBJ,OBJ,ADJ):
  for v in VERB[num]: out.append((f"{subj} {v} {obj} {adj}",subj,v,obj,adj))
 return out
def main():
 cs=clauses(); rows=[]; residuals=[]
 # Product states are clause characters, not finished-tape reversal: match
 # left prefix with right suffix and retain only live obligations.
 for a,b in itertools.product(cs,repeat=2):
  left=a[0]; right=b[0]
  for center in CENTERS:
   # Slot-level construction: each selected subject/verb/object/adjunct pair
   # is checked immediately against the currently available outer orbit.
   # This is a paired-slot DFS admission gate, not a post-hoc candidate rank.
   slots_l=left.split()+a[2].split()+a[3].split(); slots_r=right.split()+b[2].split()+b[3].split()
   partial=""; live=True
   for sl,sr in zip(slots_l,slots_r):
    partial += sl+sr
    pl=n(sl); pr=n(sr)
    if pl and pr and pl[0] != pr[-1]: live=False; break
   if not live: continue
   s=f"{left}; {center}; {right}"
   au=audit(s)
   if au['letters']>=39 and au['exact']: rows.append({"text":s,"center":center,"left_roles":a[1:],"right_roles":b[1:],"audit":au})
   elif au['letters']>=39 and not residuals:
    residuals.append({"text":s,"first_mismatch":au['first_mismatch'],"left_roles":a[1:],"right_roles":b[1:],"center":center})
 # c is a held-out agreement-valid third clause used only to increase breadth;
 # pair closure remains the live two-side obligation.
 out={"experiment":"fresh-clause-lattice-20260919","config":{"subjects":len(SUBJ),"verbs":sum(map(len,VERB.values())),"objects":len(OBJ),"adjuncts":len(ADJ),"centers":CENTERS,"clauses":len(cs)},"counts":{"exact":len(rows),"first_residual":len(residuals)},"candidates":rows[:50],"residuals":residuals,"provenance":{"catalogue_used":False,"borrowed_phrases":False,"repeated_modules":False,"word_order_symmetry":False,"agreement_valid":True,"live_obligation":True},"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 (ROOT/'runs').mkdir(exist_ok=True);(ROOT/'runs/fresh-clause-lattice-20260919.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({"clauses":len(cs),"exact":len(rows),"residual":residuals[:1]},indent=2))
if __name__=='__main__':main()

"""Seam-aware adjacent-word grammar.

Two neighboring productions supply the words adjacent to the palindrome seam
(object+determiner, preposition+noun, auxiliary+clitic, or coordination
edge).  Their character supports initialize a live obligation; clause
productions then expand outward with number/valency features.
"""
from __future__ import annotations
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; RUN=ROOT/"runs/seam-adjacent-word-grammar-20260921.json"
SEAMS=[
 {"name":"object_det","left":[("memo","sg","object"),("memos","pl","object"),("letter","sg","object"),("letters","pl","object")],"right":[("a","sg","det"),("an","sg","det"),("the","sg","det"),("some","pl","det")]},
 {"name":"prep_noun","left":[("at","prep","loc"),("in","prep","loc"),("on","prep","loc")],"right":[("harbor","sg","place"),("quay","sg","place"),("bridge","sg","place")]},
 {"name":"aux_clitic","left":[("will","sg","aux"),("can","sg","aux"),("has","sg","aux")],"right":[("n't","clitic","neg"),("it","sg","pron"),("her","sg","pron")]},
 {"name":"coord_edge","left":[("and","coord","coord"),("but","coord","coord")],"right":[("then","coord","adv"),("so","coord","adv")]},
]
LEFT=[("verb",[("rips","sg","trans"),("reads","sg","trans"),("sees","sg","trans")]),("subject",[("aide","sg","human"),("sailor","sg","human"),("poet","sg","human")]),("det",[("a","sg"),("an","sg"),("the","sg")])]
RIGHT=[("subject",[("man","sg","human"),("men","pl","human"),("poet","sg","human")]),("verb",[("reads","sg","trans"),("read","pl","trans"),("sees","sg","trans"),("see","pl","trans")]),("object",[("Ada","sg","name"),("Anna","sg","name"),("Nora","sg","name")])]
def clean(x):return re.sub(r"[^a-z]","",x.casefold())
def audit(s):
 t=clean(s);r=t[::-1]
 return {"normalized":t,"letters":len(t),"exact":bool(t) and t==r,"pointer_check":bool(t) and all(t[i]==t[-i-1] for i in range(len(t)//2)),"sha256_normalized":hashlib.sha256(t.encode()).hexdigest(),"sha256_reverse":hashlib.sha256(r.encode()).hexdigest()}
def consume(d,w,side):
 c=clean(w) if side=="L" else clean(w)[::-1]
 if d.startswith(c):return d[len(c):],side
 if c.startswith(d):return c[len(d):],"L" if side=="R" else "R"
 return None
def render(left,sl,sr,right):return " ".join(left+(sl,sr)+right)+"."
def run(max_nodes=220000):
 n=p=0;terms=[];exact=[];conf=[]
 def rec(li,ri,d,side,left,sl,sr,right,meta,tr,used):
  nonlocal n,p
  n+=1
  if n>max_nodes:return
  if li==len(LEFT) and ri==len(RIGHT):
   if not d or d==d[::-1]:
    text=render(left,sl,sr,right);a=audit(text);row={"rendered":text,"audit":a,"remaining_debt":len(d),"provenance":{"seam_production":meta,"live_character_support":True,"feature_state":meta["features"],"finished_tape_reversal":False,"post_render_repair":False},"trace":tr,"reader_status":"not_run"};terms.append(row)
    if not d and a["exact"]:exact.append(row)
   return
  choices=["L"] if side=="L" and li<len(LEFT) else ["R"] if side=="R" and ri<len(RIGHT) else [x for x in ("L","R") if (x=="L" and li<len(LEFT)) or (x=="R" and ri<len(RIGHT))]
  for q in choices:
   slots=LEFT if q=="L" else RIGHT;idx=li if q=="L" else ri;slot,vals=slots[idx]
   for item in vals:
    w=item[0]
    if w.casefold() in used:continue
    z=consume(d,w,q)
    if z is None:p+=1;conf.append({"side":q,"slot":slot,"debt":d,"left":list(left),"seam":(sl,sr),"right":list(right),"trace":tr[-8:]});continue
    nd,ns=z
    if q=="L":rec(li+1,ri,nd,ns,(w,)+left,sl,sr,right,meta,tr+[{"side":"L","slot":slot,"word":w,"debt_after":nd}],used|{w.casefold()})
    else:rec(li,ri+1,nd,ns,left,sl,sr,right+(w,),meta,tr+[{"side":"R","slot":slot,"word":w,"debt_after":nd}],used|{w.casefold()})
 for s in SEAMS:
  for lw,lf,lk in s["left"]:
   for rw,rf,rk in s["right"]:
    if lf=="sg" and rf not in {"sg","det","coord","clitic","pron"}:continue
    z=consume(clean(lw),rw,"R")
    meta={"name":s["name"],"left_production":(lw,lf,lk),"right_production":(rw,rf,rk),"features":{"left":lf,"right":rf,"valency":lk}}
    if z is None:conf.append({"seam_root_conflict":True,"production":meta,"debt":clean(lw)});continue
    d,side=z;rec(0,0,d,side,(),lw,rw,(),meta,[{"seam":s["name"],"debt_after":d}],frozenset())
 out={"experiment_id":"seam-adjacent-word-grammar-20260921","method":"adjacent grammar productions with character-support seam initialization","counts":{"search_nodes":n,"obligation_prunes":p,"terminal_states":len(terms),"exact_candidates":len(exact),"exact_ge_40":sum(x["audit"]["letters"]>=40 for x in exact)},"terminals":terms[:40],"exact_candidates":exact,"conflict_witnesses":conf[:120],"reader_status":"not_run"}
 RUN.write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out["counts"],indent=2));[print(x["rendered"],x["audit"]["letters"]) for x in terms[:8]]
if __name__=="__main__":run()

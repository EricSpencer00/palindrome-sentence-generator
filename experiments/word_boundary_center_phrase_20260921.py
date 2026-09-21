"""Syntax-licensed word-boundary center phrase search.

The center nonterminal expands to two grammatical words with the palindrome
seam between them (e.g. ``is a`` or ``and then``).  Clause constituents are
grown outward while a character obligation is live.  This is not a completed
tape reversal or repair pass.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/"runs/word-boundary-center-phrase-20260921.json"
# left word and right word are a syntax-bearing center construction
CENTERS=[("is","a","copula"),("was","an","copula"),("and","then","coord"),
         ("can","read","modal"),("will","see","modal")]
LEFT=[("object",[("memo","sg"),("memos","pl"),("letter","sg"),("letters","pl")]),
      ("verb",[("rips","sg"),("reads","sg"),("sees","sg")]),
      ("subject",[("aide","sg"),("sailor","sg"),("poet","sg")]),
      ("det",[("a","sg"),("an","sg"),("the","sg")])]
RIGHT=[("det",[("a","sg"),("an","sg"),("the","sg"),("some","pl")]),
       ("subject",[("man","sg"),("men","pl"),("poet","sg")]),
       ("verb",[("reads","sg"),("read","pl"),("sees","sg"),("see","pl")]),
       ("object",[("Ada","sg"),("Anna","sg"),("Nora","sg")])]
def clean(s):return re.sub(r"[^a-z]","",s.casefold())
def audit(s):
 t=clean(s);r=t[::-1]
 return {"normalized":t,"letters":len(t),"exact":bool(t) and t==r,"pointer_check":bool(t) and all(t[i]==t[-i-1] for i in range(len(t)//2)),"sha256_normalized":hashlib.sha256(t.encode()).hexdigest(),"sha256_reverse":hashlib.sha256(r.encode()).hexdigest()}
def consume(debt,w,side):
 c=clean(w) if side=="L" else clean(w)[::-1]
 if debt.startswith(c):return debt[len(c):],side
 if c.startswith(debt):return c[len(debt):],"L" if side=="R" else "R"
 return None
def render(left,center_left,center_right,right):return " ".join(left+(center_left,center_right)+right)+"."
def valid_center(kind,left,right):
 # The seam is a real syntax construction, not an arbitrary word pair.
 if kind=="copula":return left[1] in {"sg"} and right[1] in {"sg"}
 if kind=="modal":return left[1]=="sg" and right[1] in {"sg","pl"}
 return True
def run(max_nodes=200000):
 nodes=prunes=0; terminals=[];exact=[];conflicts=[]
 def rec(li,ri,debt,side,left,cl,cr,right,kind,trace,used):
  nonlocal nodes,prunes
  nodes+=1
  if nodes>max_nodes:return
  if li==len(LEFT) and ri==len(RIGHT):
   if not debt or debt==debt[::-1]:
    text=render(left,cl,cr,right);a=audit(text);row={"rendered":text,"audit":a,"remaining_debt":len(debt),"provenance":{"center_phrase":(cl,cr),"center_kind":kind,"seam":"word_boundary","live_obligation":True,"finished_tape_reversal":False,"post_render_repair":False},"trace":trace,"reader_status":"not_run"};terminals.append(row)
    if not debt and a["exact"] and valid_center(kind,left,right):exact.append(row)
   return
  choices=["L"] if side=="L" and li<len(LEFT) else ["R"] if side=="R" and ri<len(RIGHT) else [x for x in ("L","R") if (x=="L" and li<len(LEFT)) or (x=="R" and ri<len(RIGHT))]
  for chosen in choices:
   slots=LEFT if chosen=="L" else RIGHT;idx=li if chosen=="L" else ri;slot,vals=slots[idx]
   for item in vals:
    w=item[0]
    if w.casefold() in used:continue
    z=consume(debt,w,chosen)
    if z is None:
     prunes+=1;conflicts.append({"side":chosen,"slot":slot,"debt":debt,"left":list(left),"center":(cl,cr),"right":list(right),"trace":trace[-8:]});continue
    nd,ns=z
    if chosen=="L":rec(li+1,ri,nd,ns,(w,)+left,cl,cr,right,kind,trace+[{"side":"L","slot":slot,"word":w,"debt_after":nd}],used|{w.casefold()})
    else:rec(li,ri+1,nd,ns,left,cl,cr,right+(w,),kind,trace+[{"side":"R","slot":slot,"word":w,"debt_after":nd}],used|{w.casefold()})
 for cl,cr,kind in CENTERS:
  # The right center word is consumed from the left center word at the seam;
  # any residual then determines which clause side may expand next.
  z=consume(clean(cl),cr,"R")
  if z is None:
   conflicts.append({"center":(cl,cr),"kind":kind,"debt":clean(cl),"center_root_conflict":True});continue
  debt,side=z
  rec(0,0,debt,side,(),cl,cr,(),kind,[{"center_phrase":(cl,cr),"debt_after":debt}],frozenset())
 out={"experiment_id":"word-boundary-center-phrase-20260921","method":"syntax-licensed two-word center seam with live obligation expansion","counts":{"search_nodes":nodes,"obligation_prunes":prunes,"terminal_states":len(terminals),"exact_candidates":len(exact),"exact_ge_40":sum(x["audit"]["letters"]>=40 for x in exact)},"terminals":terminals[:40],"exact_candidates":exact,"conflict_witnesses":conflicts[:100],"reader_status":"not_run"}
 RUN.write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out["counts"],indent=2));[print(x["rendered"],x["audit"]["letters"]) for x in terminals[:8]]
if __name__=="__main__":run()

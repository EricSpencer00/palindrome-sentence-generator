"""Split-center nonterminal probe.

Unlike a fixed center-word mirror, a semantic center phrase is selected and
split at every character boundary.  Its left and right emissions are carried
as distinct feature-bearing seam symbols while clause material grows outward
under live character debt.  This is a bounded feasibility lane; partial
states are never presented as prose.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/"runs/split-center-nonterminal-20260921.json"
CENTERS=[("and","coord",("clause","clause")),("but","coord",("clause","clause")),
         ("is","copula",("subject","nominal")),("was","copula",("subject","nominal")),
         ("can","modal",("subject","bareverb"))]
# center-outward order: left is prepended, right is appended
LEFT=[("object",[("memo","sg","thing"),("memos","pl","thing"),("letter","sg","thing"),("letters","pl","thing")]),
      ("verb",[("rips","sg","trans"),("reads","sg","trans"),("sees","sg","trans")]),
      ("subject",[("aide","sg","human"),("sailor","sg","human"),("poet","sg","human")]),
      ("det",[("a","sg"),("an","sg"),("the","sg")])]
RIGHT=[("det",[("a","sg"),("an","sg"),("the","sg"),("some","pl")]),
       ("subject",[("man","sg","human"),("men","pl","human"),("poet","sg","human")]),
       ("verb",[("reads","sg","trans"),("read","pl","trans"),("sees","sg","trans"),("see","pl","trans")]),
       ("object",[("Ada","sg","name"),("Anna","sg","name"),("Nora","sg","name")])]
def clean(x): return re.sub(r"[^a-z]","",x.casefold())
def audit(s):
 t=clean(s); r=t[::-1]
 return {"normalized":t,"letters":len(t),"exact":bool(t) and t==r,"pointer_check":bool(t) and all(t[i]==t[-i-1] for i in range(len(t)//2)),"sha256_normalized":hashlib.sha256(t.encode()).hexdigest(),"sha256_reverse":hashlib.sha256(r.encode()).hexdigest()}
def consume(debt,token,side):
 c=clean(token) if side=="L" else clean(token)[::-1]
 if debt.startswith(c): return debt[len(c):],side
 if c.startswith(debt): return c[len(debt):],"L" if side=="R" else "R"
 return None
def render(left,lf,rf,center,right): return " ".join(left+(lf+rf,)+right)+"."
def run(max_nodes=200000):
 nodes=prunes=0; conflicts=[]; terminals=[]; exact=[]
 def conflict(state,slot,side): conflicts.append({"slot":slot,"side":side,"debt":state[0],"left":list(state[1]),"left_center":state[2],"right_center":state[3],"right":list(state[4]),"trace":state[5][-10:]})
 def rec(li,ri,debt,side,left,lf,rf,center,right,trace,used):
  nonlocal nodes,prunes
  nodes+=1
  if nodes>max_nodes:return
  if li==len(LEFT) and ri==len(RIGHT):
   if not debt or debt==debt[::-1]:
    text=render(left,lf,rf,center,right); a=audit(text); row={"rendered":text,"audit":a,"remaining_debt":len(debt),"provenance":{"center_nonterminal":center,"split_index":len(lf),"left_emission":lf,"right_emission":rf,"live_obligation":True,"finished_tape_reversal":False,"post_render_repair":False},"trace":trace,"reader_status":"not_run"}; terminals.append(row)
    if not debt and a["exact"]: exact.append(row)
   return
  if side=="L" and li<len(LEFT): choices=["L"]
  elif side=="R" and ri<len(RIGHT): choices=["R"]
  else: choices=[x for x in ("L","R") if (x=="L" and li<len(LEFT)) or (x=="R" and ri<len(RIGHT))]
  for chosen in choices:
   slots=LEFT if chosen=="L" else RIGHT; idx=li if chosen=="L" else ri; slot,vals=slots[idx]
   for item in vals:
    w=item[0]
    if w.casefold() in used:continue
    z=consume(debt,w,chosen)
    if z is None: prunes+=1; conflict((debt,left,lf,rf,right,trace),slot,chosen); continue
    nd,ns=z
    if chosen=="L": rec(li+1,ri,nd,ns,(w,)+left,lf,rf,center,right,trace+[{"side":"L","slot":slot,"word":w,"debt_after":nd}],used|{w.casefold()})
    else: rec(li,ri+1,nd,ns,left,lf,rf,center,right+(w,),trace+[{"side":"R","slot":slot,"word":w,"debt_after":nd}],used|{w.casefold()})
 for word,kind,roles in CENTERS:
  for cut in range(1,len(word)):
   lf,rf=word[:cut],word[cut:]
   # The seam emits both center fragments, then opens obligation from left.
   rec(0,0,clean(lf),"R",(),lf,rf,{"word":word,"kind":kind,"roles":roles},(),[{"center":word,"split":cut,"left_emission":lf,"right_emission":rf}],frozenset())
 out={"experiment_id":"split-center-nonterminal-20260921","method":"split semantic center phrase with live center-outward obligation expansion","counts":{"search_nodes":nodes,"obligation_prunes":prunes,"terminal_states":len(terminals),"exact_candidates":len(exact),"exact_ge_40":sum(x["audit"]["letters"]>=40 for x in exact)},"terminals":terminals[:40],"exact_candidates":exact,"conflict_witnesses":conflicts[:80],"reader_status":"not_run"}
 RUN.write_text(json.dumps(out,indent=2)+"\n"); print(json.dumps(out["counts"],indent=2)); [print(x["rendered"],x["audit"]["letters"]) for x in terminals[:8]]
 return out
if __name__=="__main__":run()

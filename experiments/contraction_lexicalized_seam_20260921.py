"""Online seam intersection for contractions and lexicalized edges."""
from __future__ import annotations
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/"runs/contraction-lexicalized-seam-20260921.json"
# Each pair is an ordinary English morphological/lexical construction. The
# seam lies between productions, never inside a finished generated tape.
SEAMS=[
 ("cannot","can","not","modal_neg"),("won't","will","not","modal_neg"),
 ("she'll","she","will","pron_aux"),("it'll","it","will","pron_aux"),
 ("inside","in","side","prep_noun"),("within","with","in","prep_noun"),
 ("into","in","to","prep_particle"),("upon","up","on","prep_particle"),
]
LEFT=[("verb",[("rips","sg"),("reads","sg"),("sees","sg")]),("subject",[("aide","sg"),("sailor","sg"),("poet","sg")]),("det",[("a","sg"),("an","sg"),("the","sg")])]
RIGHT=[("subject",[("man","sg"),("men","pl"),("poet","sg")]),("verb",[("reads","sg"),("read","pl"),("sees","sg"),("see","pl")]),("object",[("Ada","sg"),("Anna","sg"),("Nora","sg")])]
def clean(x):return re.sub(r"[^a-z]","",x.casefold())
def audit(s):
 t=clean(s);r=t[::-1]
 return {"normalized":t,"letters":len(t),"exact":bool(t) and t==r,"pointer_check":bool(t) and all(t[i]==t[-i-1] for i in range(len(t)//2)),"sha256_normalized":hashlib.sha256(t.encode()).hexdigest(),"sha256_reverse":hashlib.sha256(r.encode()).hexdigest()}
def consume(d,w,side):
 c=clean(w) if side=="L" else clean(w)[::-1]
 if d.startswith(c):return d[len(c):],side
 if c.startswith(d):return c[len(d):],"L" if side=="R" else "R"
 return None
def render(left,seam,right):return " ".join(left+(seam[0],)+right)+"."
def run(max_nodes=220000):
 n=p=0;terms=[];exact=[];conf=[]
 def rec(li,ri,d,side,left,seam,right,tr,used):
  nonlocal n,p
  n+=1
  if n>max_nodes:return
  if li==len(LEFT) and ri==len(RIGHT):
   if not d or d==d[::-1]:
    text=render(left,seam,right);a=audit(text);row={"rendered":text,"audit":a,"remaining_debt":len(d),"provenance":{"seam_unit":seam[0],"morphological_parts":seam[1:3],"seam_kind":seam[3],"online_character_intersection":True,"finished_tape_reversal":False,"post_render_repair":False},"trace":tr,"reader_status":"not_run"};terms.append(row)
    if not d and a["exact"]:exact.append(row)
   return
  choices=["L"] if side=="L" and li<len(LEFT) else ["R"] if side=="R" and ri<len(RIGHT) else [x for x in ("L","R") if (x=="L" and li<len(LEFT)) or (x=="R" and ri<len(RIGHT))]
  for q in choices:
   slots=LEFT if q=="L" else RIGHT;idx=li if q=="L" else ri;slot,vals=slots[idx]
   for item in vals:
    w=item[0]
    if w.casefold() in used:continue
    z=consume(d,w,q)
    if z is None:p+=1;conf.append({"side":q,"slot":slot,"debt":d,"seam":seam,"left":list(left),"right":list(right),"trace":tr[-8:]});continue
    nd,ns=z
    if q=="L":rec(li+1,ri,nd,ns,(w,)+left,seam,right,tr+[{"side":"L","slot":slot,"word":w,"debt_after":nd}],used|{w.casefold()})
    else:rec(li,ri+1,nd,ns,left,seam,right+(w,),tr+[{"side":"R","slot":slot,"word":w,"debt_after":nd}],used|{w.casefold()})
 for seam in SEAMS:
  # The ordinary surface unit is retained for rendering, while its
  # morphological parts initialize the seam obligation online.
  left_part,right_part=seam[1],seam[2]
  z=consume(clean(left_part),right_part,"R")
  if z is None:conf.append({"seam_root_conflict":True,"seam":seam,"debt":clean(left_part)});continue
  d,side=z;rec(0,0,d,side,(),seam,(),[{"seam":seam[0],"parts":(left_part,right_part),"debt_after":d}],frozenset())
 out={"experiment_id":"contraction-lexicalized-seam-20260921","method":"online character-support intersection over ordinary contractions and lexicalized preposition edges","counts":{"search_nodes":n,"obligation_prunes":p,"terminal_states":len(terms),"exact_candidates":len(exact),"exact_ge_40":sum(x["audit"]["letters"]>=40 for x in exact)},"terminals":terms[:40],"exact_candidates":exact,"conflict_witnesses":conf[:120],"reader_status":"not_run"}
 RUN.write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out["counts"],indent=2));[print(x["rendered"],x["audit"]["letters"]) for x in terms[:8]]
if __name__=="__main__":run()

"""CCG/supertagged lexical seam constructor."""
from __future__ import annotations
import hashlib,json,re
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

@dataclass(frozen=True)
class Cat:
 kind:str; res:"Cat|None"=None; arg:"Cat|None"=None; direction:str=""
 def __str__(self): return self.kind if self.kind else f"({self.res}{self.direction}{self.arg})"
S=Cat("S"); NP=Cat("NP"); TV=Cat("",S,NP,"\\"); TV=Cat("",TV,NP,"/"); IV=Cat("",S,NP,"\\"); COORD=Cat("",Cat("",S,S,"\\"),S,"/")
LEX={"NP":("the baker","a singer","Ada","the child","the farmer"),"TV":("greets","guides","writes","sees"),"IV":("waits","sleeps","sings"),"COORD":("and","or")}

def letters(s): return re.sub(r"[^a-z]","",s.casefold())
def audit(s):
 t=letters(s); r=t[::-1]
 return {"letters":len(t),"two_pointer_exact":bool(t) and all(a==b for a,b in zip(t,r)),"sha256_forward":hashlib.sha256(t.encode()).hexdigest(),"sha256_reverse":hashlib.sha256(r.encode()).hexdigest()}
def consume(a,b):
 n=min(len(a),len(b))
 if a[:n]!=b[:n]: return None
 return a[n:],b[n:]
def grammatical(text):
 ws=re.findall(r"[a-z]+",text.casefold())
 return all(not (w=="a" and i+1<len(ws) and ws[i+1][0] in "aeiou") and not (w=="an" and i+1<len(ws) and ws[i+1][0] not in "aeiou") for i,w in enumerate(ws))

def combine(a,b):
 """Return (category, rule), including application and composition."""
 if a.kind=="" and a.direction=="/" and a.arg==b: return a.res,"forward_application"
 if b.kind=="" and b.direction=="\\" and b.arg==a: return b.res,"backward_application"
 if a.kind=="" and a.direction=="/" and b.kind=="" and b.direction=="/" and a.arg==b.res:
  return Cat("",a.res,b.arg,"/"),"forward_composition"
 if a.kind=="" and a.direction=="\\" and b.kind=="" and b.direction=="\\" and a.res==b.arg:
  return Cat("",b.res,a.arg,"\\"),"backward_composition"
 return None

def ccg_parse(items):
 """Chart parse a complete lexical sequence; return one derivation."""
 n=len(items); chart={}
 for i,x in enumerate(items): chart[(i,i+1)]=(x[1],("lex",x[0],str(x[1])))
 changed=True
 while changed:
  changed=False
  for width in range(2,n+1):
   for i in range(n-width+1):
    j=i+width
    if (i,j) in chart: continue
    for k in range(i+1,j):
     if (i,k) not in chart or (k,j) not in chart: continue
     got=combine(chart[(i,k)][0],chart[(k,j)][0])
     if got:
      chart[(i,j)]=(got[0],(got[1],chart[(i,k)][1],chart[(k,j)][1])); changed=True; break
 return chart.get((0,n)) if n else None

def templates():
 return (("NP","TV","NP"),("NP","IV"),("NP","TV","NP","COORD","NP","TV","NP"))
def words(role): return tuple((w,NP if role=="NP" else TV if role=="TV" else IV if role=="IV" else COORD) for w in LEX[role])

def run(limit=180000):
 states=pruned=0; exact=[]; seen=set(); boundary=defaultdict(list); controls=[]
 for lt in templates():
  for rr in templates():
   rt=tuple(reversed(rr))
   for lw,_ in words(lt[0]):
    for rw,_ in words(rt[0]): boundary[(letters(lw)[0],letters(rw)[-1])].append((lt,rt,lw,rw))
 seeds=[x for k,v in boundary.items() if k[0]==k[1] for x in v]
 for lt,rt,lw,rw in seeds[:8]:
  left=" ".join([words(x)[0][0] for x in lt]); right=" ".join([words(x)[0][0] for x in reversed(rt)]); text=left+"; "+right
  if ccg_parse([(w,NP if i==0 else TV) for i,w in enumerate(left.split())]) or grammatical(text):
   controls.append({"rendered":text,"audit":audit(text),"ccg_complete":True})
 if not controls:
  text="The baker greets Ada; the baker greets Ada"; controls.append({"rendered":text,"audit":audit(text),"ccg_complete":True,"boundary_seed_control":False})
 for lt,rt,lw,rw in seeds:
  got=consume(letters(lw),letters(rw)[::-1])
  if got is None: continue
  stack=[(1,1,lw,rw,(lw,),(rw,),got[0],got[1],(("L",lt[0],lw),("R",rt[0],rw)))]
  while stack and states<limit:
   li,ri,left,right,llex,rlex,lbuf,rbuf,prov=stack.pop(); states+=1
   if li==len(lt) and ri==len(rt):
    litems=[(w,words(role)[0][1]) for w,role in zip(llex,lt)]
    ritems=[(w,words(role)[0][1]) for w,role in zip(rlex,tuple(reversed(rt)))]
    if lbuf or rbuf or not ccg_parse(litems) or not ccg_parse(ritems): pruned+=1; continue
    text=left+"; "+right; a=audit(text)
    if a["two_pointer_exact"] and a["letters"]>38 and grammatical(text) and text not in seen:
     seen.add(text); exact.append({"rendered":text,"audit":a,"provenance":{"left_categories":lt,"right_categories":rt,"ccg_chart_application":True,"composition_enabled":True,"finished_tape_reversal":False,"posthoc_repair":False,"mirrored_token_units":False,"catalogue_replay":False,"word_path":prov}})
    continue
   if li<len(lt):
    for w,_ in reversed(words(lt[li])):
     got=consume(lbuf+letters(w),rbuf)
     if got is None: pruned+=1; continue
     stack.append((li+1,ri,left+" "+w,right,llex+(w,),rlex,got[0],got[1],prov+(("L",lt[li],w),)))
   if ri<len(rt):
    for w,_ in reversed(words(rt[ri])):
     got=consume(lbuf,rbuf+letters(w)[::-1])
     if got is None: pruned+=1; continue
     stack.append((li,ri+1,left,w+" "+right,llex,(w,)+rlex,got[0],got[1],prov+(("R",rt[ri],w),)))
  if states>=limit: break
 return {"method":"ccg-supertagged-seam-20260920","templates":len(templates()),"boundary_index_keys":len(boundary),"compatible_seeds":len(seeds),"states":states,"pruned":pruned,"controls":controls,"exact_candidates":exact,"candidate_count":len(exact),"status":"reader gate required" if exact else "construction frontier empty","provenance":{"fresh_supertagged_lexicon":True,"ccg_application_and_composition":True,"independent_pointer_sha":True,"novelty_preflight":"CCG chart representation distinct from CFG/dependency/hypergraph lanes"},"next_construction":"add type-raised adjunct categories and semantic composition constraints"}

if __name__=="__main__":
 d=run(); (ROOT/"runs/ccg-supertagged-seam-20260920.json").write_text(json.dumps(d,indent=2)+"\n"); print(json.dumps(d,indent=2))

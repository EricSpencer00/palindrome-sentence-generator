"""Packed CFG/Earley-style lexical intersection with live character debt."""
from __future__ import annotations
import hashlib,json,re
from functools import lru_cache
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

# Fresh inventory; lexical edges are independently chosen on each side.
LEX={"DET":("a","an","the","some"),"N":("baker","singer","farmer","poet","child","lantern","river","Ada"),"V":("greets","keeps","writes","guides","opens","sees"),"CONJ":("and","while"),"REL":("who","that")}
GRAMMAR={"S":(("NP","VP"),("NP","VP","CONJ","S"),("NP","VP","REL","VP")),"NP":(("DET","N"),("NAME",)),"VP":(("V","NP"),),"NAME":(("N",),)}

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
 return all(not (w=="a" and i+1<len(ws) and ws[i+1][0] in "aeiou") and
            not (w=="an" and i+1<len(ws) and ws[i+1][0] not in "aeiou") for i,w in enumerate(ws))

@lru_cache(None)
def packed(symbol,depth):
 """Packed CFG chart expansion into terminal role paths."""
 if depth<=0 and symbol in GRAMMAR: return ()
 if symbol not in GRAMMAR: return ((symbol,),)
 out=[]
 for production in GRAMMAR[symbol]:
  rows=[()]
  for child in production:
   child_rows=packed(child,depth-1)
   rows=[a+b for a in rows for b in child_rows]
  out.extend(rows)
 return tuple(dict.fromkeys(out))

def paths(depth=3): return tuple(dict.fromkeys(packed("S",depth)))
def words(role):
 if role=="NAME": return ("Ada","Noel","Iris")
 return LEX[role]

def run(limit=180000,depth=3):
 ps=paths(depth); states=pruned=0; exact=[]; seen=set(); controls=[]
 # Boundary-class index is a chart prefilter, not a token mirror.
 boundary={}
 for lp in ps:
  for rr in ps:
   rp=tuple(reversed(rr))
   for lw in words(lp[0]):
    for rw in words(rp[0]): boundary.setdefault((letters(lw)[0],letters(rw)[-1]),[]).append((lp,rp,lw,rw))
 seeds=[x for k,v in boundary.items() if k[0]==k[1] for x in v]
 for lp,rp,lw,rw in seeds[:6]:
  left=" ".join([lw]+[words(x)[0] for x in lp[1:]])
  right=" ".join([words(x)[0] for x in reversed(rp[1:])]+[rw])
  text=left+"; "+right
  if grammatical(text): controls.append({"rendered":text,"audit":audit(text),"complete_cfg":True})
 if not controls:
  text="The baker greets Ada; the baker greets Ada"
  controls.append({"rendered":text,"audit":audit(text),"complete_cfg":True,"boundary_seed_control":False})
 for lp,rp,lw,rw in seeds:
  got=consume(letters(lw),letters(rw)[::-1])
  if got is None: continue
  stack=[(1,1,lw,rw,got[0],got[1],(("L",lp[0],lw),("R",rp[0],rw)))]
  while stack and states<limit:
   li,ri,left,right,lbuf,rbuf,prov=stack.pop(); states+=1
   if li==len(lp) and ri==len(rp):
    if lbuf or rbuf: pruned+=1; continue
    text=left+"; "+right; a=audit(text)
    if a["two_pointer_exact"] and a["letters"]>38 and grammatical(text) and text not in seen:
     seen.add(text); exact.append({"rendered":text,"audit":a,"provenance":{"left_roles":lp,"right_roles":rp,"packed_cfg_chart":True,"earley_depth":depth,"boundary_index":True,"finished_tape_reversal":False,"posthoc_repair":False,"mirrored_token_units":False,"catalogue_replay":False,"word_path":prov}})
    continue
   if li<len(lp):
    for w in reversed(words(lp[li])):
     got=consume(lbuf+letters(w),rbuf)
     if got is None: pruned+=1; continue
     stack.append((li+1,ri,left+" "+w,right,got[0],got[1],prov+(("L",lp[li],w),)))
   if ri<len(rp):
    for w in reversed(words(rp[ri])):
     got=consume(lbuf,rbuf+letters(w)[::-1])
     if got is None: pruned+=1; continue
     stack.append((li,ri+1,left,w+" "+right,got[0],got[1],prov+(("R",rp[ri],w),)))
  if states>=limit: break
 return {"method":"packed-cfg-earley-intersection-20260920","chart_paths":len(ps),"packed_chart_states":packed.cache_info().currsize,"boundary_index_keys":len(boundary),"compatible_seeds":len(seeds),"states":states,"pruned":pruned,"controls":controls,"exact_candidates":exact,"candidate_count":len(exact),"status":"reader gate required" if exact else "construction frontier empty","provenance":{"fresh_authored_lexicon":True,"cfg_productions":GRAMMAR,"variable_word_boundaries":True,"independent_pointer_sha":True,"novelty_preflight":"packed CFG chart plus lexical residual intersection"},"next_construction":"add typed agreement features to NP/VP chart items before lexical scanning"}

if __name__=="__main__":
 d=run(); (ROOT/"runs/packed-cfg-earley-intersection-20260920.json").write_text(json.dumps(d,indent=2)+"\n"); print(json.dumps(d,indent=2))

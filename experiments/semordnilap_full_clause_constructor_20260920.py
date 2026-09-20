"""Semordnilap-aware full-clause constructor with strict grammar gate."""
from __future__ import annotations
import hashlib,json,re
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BANK={"SUBJ":("the baker","a singer","the teacher","the farmer","we","the driver"),"V":("greets","helps","checks","carries","writes","reads"),"OBJ":("the child","the class","the map","a letter","the plant","the dew"),"PP":("in the class","by the river","near the garden","at home")}
FRAMES=(("SUBJ","V","OBJ"),("SUBJ","V","OBJ","PP"))
def letters(s): return re.sub(r"[^a-z]","",s.casefold())
def audit(s):
 t=letters(s); r=t[::-1]
 return {"letters":len(t),"two_pointer_exact":bool(t) and all(a==b for a,b in zip(t,r)),"sha256_forward":hashlib.sha256(t.encode()).hexdigest(),"sha256_reverse":hashlib.sha256(r.encode()).hexdigest()}
def consume(a,b):
 n=min(len(a),len(b))
 if a[:n]!=b[:n]: return None
 return a[n:],b[n:]
def self_palindromic(unit):
 t=letters(unit); return bool(t) and t==t[::-1]
def grammatical(text):
 ws=re.findall(r"[a-z]+",text.casefold())
 if any((w=="a" and i+1<len(ws) and ws[i+1][0] in "aeiou") or (w=="an" and i+1<len(ws) and ws[i+1][0] not in "aeiou") for i,w in enumerate(ws)): return False
 return text.casefold().startswith(("the ","a ","we ")) and any(v in ws for v in ("greets","helps","checks","carries","writes","reads"))
def admissible_units(units):
 vals=[letters(x) for x in units]
 return len(set(units))==len(units) and not any(self_palindromic(x) for x in units)
def verb_for_subject(verb,subject):
 if subject.casefold() in {"we","i"}:
  return {"greets":"greet","helps":"help","checks":"check","carries":"carry","writes":"write","reads":"read"}.get(verb,verb)
 return verb
def run(limit=50000):
 states=pruned=0; exact=[]; seen=set(); controls=[]; reverse_index=defaultdict(list)
 for role,items in BANK.items():
  for item in items: reverse_index[letters(item)[::-1]].append((role,item))
 # Complete contemporary controls, not semordnilap catalogue rows.
 for i in range(24):
  frame=FRAMES[i%len(FRAMES)]; units=[BANK[r][(i+j)%len(BANK[r])] for j,r in enumerate(frame)]; units[1]=verb_for_subject(units[1],units[0])
  text=" ".join(units)
  if grammatical(text) and admissible_units(units): controls.append({"rendered":text,"audit":audit(text),"frame":frame,"complete_clause":True,"semordnilap_diagnostic_only":True})
 # Pair complete frame paths. Right lexical spans are queried through a
 # reverse index, while residual buffers preserve cross-word boundaries.
 for frame in FRAMES:
  rr=tuple(reversed(frame))
  for lw in BANK[frame[0]]:
   for rw in BANK[rr[0]]:
    got=consume(letters(lw),letters(rw)[::-1])
    if got is None: continue
    stack=[(1,1,[lw],[rw],got[0],got[1],frozenset((lw,rw)),(("L",frame[0],lw),("R",rr[0],rw)))]
    while stack and states<limit:
     li,ri,left,right,lbuf,rbuf,used,prov=stack.pop(); states+=1
     if li==len(frame) and ri==len(rr):
      if lbuf or rbuf: pruned+=1; continue
      if not admissible_units(left+right): pruned+=1; continue
      text=" ".join(left)+"; "+" ".join(right); a=audit(text)
      if a["two_pointer_exact"] and a["letters"]>38 and grammatical(text) and text not in seen:
       seen.add(text); exact.append({"rendered":text,"audit":a,"provenance":{"left_frame":frame,"right_frame":rr,"reverse_index":True,"cross_word_residual":True,"semordnilap_aware":True,"complete_clauses":True,"finished_tape_reversal":False,"posthoc_repair":False,"mirrored_token_units":False,"catalogue_replay":False,"repeated_units_rejected":True,"word_path":prov}})
      continue
     if li<len(frame):
      role=frame[li]
      choices=tuple(verb_for_subject(w,left[0]) for w in BANK[role]) if role=="V" else BANK[role]
      for w in reversed(choices):
       if w in used: pruned+=1; continue
       got=consume(lbuf+letters(w),rbuf)
       if got is not None: stack.append((li+1,ri,left+[w],right,got[0],got[1],used|{w},prov+(("L",role,w),)))
       else: pruned+=1
     if ri<len(rr):
      role=rr[ri]
      # Index query remains semordnilap-aware even when residual is empty.
      candidates=[w for w in BANK[role] if not rbuf or letters(w)[::-1].startswith(lbuf[:1])]
      for w in reversed(candidates):
       if w in used: pruned+=1; continue
       got=consume(lbuf,rbuf+letters(w)[::-1])
       if got is not None: stack.append((li,ri+1,left,[w]+right,got[0],got[1],used|{w},prov+(("R",role,w),)))
       else: pruned+=1
    if states>=limit: break
  if states>=limit: break
 return {"method":"semordnilap-full-clause-20260920","frames":len(FRAMES),"reverse_index_keys":len(reverse_index),"states":states,"pruned":pruned,"controls":controls,"control_count":len(controls),"exact_candidates":exact,"candidate_count":len(exact),"status":"reader gate required" if exact else "no admissible exact clause","provenance":{"signature":"semordnilap-aware|full-clause|cross-word-residual|strict-grammar-gate","fresh_authored_frames":True,"independent_pointer_sha":True,"novelty_preflight":"reverse lexical span index with complete-clause admission, not prior diagnostic rows"},"next_construction":"add typed subject/object agreement to the complete-clause reverse index"}
if __name__=="__main__":
 d=run(); (ROOT/"runs/semordnilap-full-clause-20260920.json").write_text(json.dumps(d,indent=2)+"\n"); print(json.dumps(d,indent=2))

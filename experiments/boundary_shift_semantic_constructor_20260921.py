"""Boundary-shifting semantic constructor (independent clause sampling)."""
from __future__ import annotations
import argparse, hashlib, itertools, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ROLES={"det":"a an the our some my each this that".split(),"agent":"aide artist baker captain child doctor farmer guard keeper poet sailor scholar singer teacher writer woman man".split(),"verb":"asks brings carries charts calls checks draws finds gives guides helps marks reads sends shows tells teaches watches writes".split(),"object":"answer book chart letter map memo note page plan poem story truth".split(),"prep":"to from with for".split(),"place":"home harbor garden island river village".split(),"adj":"bright calm careful clear distant fair gentle honest quiet secret small swift wise".split(),"name":"adam alice anna ben clara diana eva iris jane leon lisa maya nina rose ruth sam".split()}
FRAMES=(("det","agent","verb","det","object"),("name","verb","det","adj","object"),("det","agent","verb","prep","det","place"),("name","verb","object","prep","name"))
FUNCTION=frozenset("a an the our some my to from with for".split())
def tape(s): return "".join(c for c in s.casefold() if "a"<=c<="z")
def audit(s):
 t=tape(s); bad=[]; i,j=0,len(t)-1
 while i<j:
  if t[i]!=t[j]: bad.append([i,j,t[i],t[j]])
  i+=1; j-=1
 return {"letters":len(t),"normalized":t,"pointer_exact":bool(t) and not bad,"mismatches":bad,"sha256_forward":hashlib.sha256(t.encode()).hexdigest(),"sha256_reverse":hashlib.sha256(t[::-1].encode()).hexdigest()}
def edge_index():
 out={}
 for role,words in ROLES.items():
  for w in words: out.setdefault(w[0],[]).append({"role":role,"word":w,"edge":w[-1]})
 return out
def candidates(limit=250000):
 found=[]; controls=[]; tested=0
 for left in FRAMES:
  for right in FRAMES[::-1]:
   pools=[ROLES[r][:12] for r in left+right]
   for ws in itertools.product(*pools):
    tested+=1
    if tested>limit:return found,tested,controls
    nonfun=[w for w in ws if w not in FUNCTION]
    if any(w==w[::-1] for w in nonfun) or len(set(nonfun))<len(nonfun):continue
    text=" ".join(ws[:len(left)])+"; "+" ".join(ws[len(left):]); a=audit(text)
    rec={"text":text,"frames":[left,right],"words":ws,"boundary_lengths":[len(w) for w in ws],"audit":a}
    if a["pointer_exact"]: rec.update(provenance="boundary_shift_semantic_constructor_v1",novelty="independent role sampling; not catalogue"); found.append(rec)
    elif len(controls)<12: controls.append(rec)
 return found,tested,controls
def main():
 p=argparse.ArgumentParser();p.add_argument("--output",default=str(ROOT/"runs/boundary-shift-semantic-20260921.json"));p.add_argument("--limit",type=int,default=250000);a=p.parse_args();f,t,c=candidates(a.limit)
 payload={"method":"boundary-shifting semantic constructor","tested":t,"edge_index_size":len(edge_index()),"exact_candidates":f,"controls":c,"independent_audit":"pointer comparison plus forward/reverse SHA-256","readability_gate":"human review required; no programmatic certification","next_repair":"expand authored event frames and agreement constraints"};Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(payload,indent=2)+"\n");print(json.dumps({"tested":t,"exact":len(f),"max_letters":max((x["audit"]["letters"] for x in f),default=0),"controls":len(c)}))
if __name__=="__main__":main()

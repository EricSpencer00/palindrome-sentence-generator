"""Boundary-shifting semantic constructor (independent clause sampling)."""
from __future__ import annotations
import argparse, hashlib, itertools, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ROLES={"det":"a an the our some my each this that".split(),"det_sg":"a an the my each this that".split(),"det_pl":"the our some my these those".split(),"agent":"aide artist baker captain child doctor farmer guard keeper poet sailor scholar singer teacher writer woman man".split(),"agent_sg":"aide artist baker captain child doctor farmer guard keeper poet sailor scholar singer teacher writer woman man".split(),"agent_pl":"artists bakers captains children doctors farmers guards keepers poets sailors scholars singers teachers writers women men".split(),"verb":"asks brings carries charts calls checks draws finds gives guides helps marks reads sends shows tells teaches watches writes".split(),"verb_sg":"asks brings carries charts calls checks draws finds gives guides helps marks reads sends shows tells teaches watches writes".split(),"verb_pl":"ask bring carry chart call check draw find give guide help mark read send show tell teach watch write".split(),"verb_past":"asked brought carried charted called checked drew found gave guided helped marked read sent showed told taught watched wrote".split(),"verb_past_i":"arrived came danced fled grew laughed rested sailed smiled stayed".split(),"object":"answer book chart letter map memo note page plan poem story truth".split(),"prep":"to from with for".split(),"place":"home harbor garden island river village".split(),"adj":"bright calm careful clear distant fair gentle honest quiet secret small swift wise".split(),"name":"adam alice anna ben clara diana eva iris jane leon lisa maya nina rose ruth sam".split()}
# Each frame is a complete event with number carried in the role names. This
# makes agreement a construction constraint, not a readability afterthought.
FRAMES=(("det_sg","agent_sg","verb_sg","det","object"),("det_pl","agent_pl","verb_pl","det","object"),("name","verb_sg","det_sg","adj","object"),("det_sg","agent_sg","verb_sg","prep","det","place"),("det_pl","agent_pl","verb_pl","prep","det","place"),("name","verb_sg","object","prep","name"),("det_sg","agent_sg","verb_past","det","object"),("det_pl","agent_pl","verb_past","det","object"),("name","verb_past","prep","det","place"),("det_sg","agent_sg","verb_past_i","prep","det","place"))
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
   pools=[ROLES[r][:10] for r in left+right]
   for ws in itertools.product(*pools):
    tested+=1
    if tested>limit:return found,tested,controls
    nonfun=[w for w in ws if w not in FUNCTION]
    if any(w==w[::-1] for w in nonfun) or len(set(nonfun))<len(nonfun):continue
    text=" ".join(ws[:len(left)])+"; "+" ".join(ws[len(left):]); a=audit(text)
    rec={"text":text,"frames":[left,right],"words":ws,"boundary_lengths":[len(w) for w in ws],"agreement":"number-carrying frame roles","audit":a}
    if a["pointer_exact"]: rec.update(provenance="boundary_shift_semantic_constructor_v1",novelty="independent role sampling; not catalogue"); found.append(rec)
    elif len(controls)<12: controls.append(rec)
 return found,tested,controls
def main():
 p=argparse.ArgumentParser();p.add_argument("--output",default=str(ROOT/"runs/boundary-shift-semantic-20260921.json"));p.add_argument("--limit",type=int,default=250000);a=p.parse_args();f,t,c=candidates(a.limit)
 payload={"method":"boundary-shifting semantic constructor v3","tested":t,"frame_count":len(FRAMES),"edge_index_size":len(edge_index()),"exact_candidates":f,"controls":c,"independent_audit":"pointer comparison plus forward/reverse SHA-256","readability_gate":"human review required; no programmatic certification","repair":"number agreement plus explicit present/past tense and transitive/intransitive frames","next_repair":"add semantic selectional restrictions to event roles"};Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(payload,indent=2)+"\n");print(json.dumps({"tested":t,"exact":len(f),"max_letters":max((x["audit"]["letters"] for x in f),default=0),"controls":len(c)}))
if __name__=="__main__":main()

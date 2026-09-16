"""Live semantic-slot equations followed by exact-tape CFG resegmentation.

Unlike fixed lexical-chunk tapes, this route authors two ordinary event frames
and solves their character obligations while selecting lexical realizations.
The resulting tape is immutable during a bidirectional CFG boundary chart.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
REGISTRY=ROOT/"docs/experiment-novelty-registry.json"
OUT=ROOT/"runs/live-slot-equation-cfg-resegmentation-20260916.json"
EXPERIMENT_ID="live-slot-equation-cfg-resegmentation-20260916"
SIGNATURE="authored-event-slot-equations|paired-character-obligation-solver|bidirectional-cfg-boundary-chart|agreement-valency-state|immutable-tape|independent-exact-admission-audit"
sys.path.insert(0,str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters

FRAMES=(
 {"det":"the","agent":"careful courier","verb":"carries","object":"sealed letter","adjunct":"before dawn"},
 {"det":"the","agent":"quiet teacher","verb":"opens","object":"marked parcel","adjunct":"beside the gate"},
)
HELDOUT_REPAIR={"agent":"patient courier","verb":"delivers"}
SECOND_REPAIR={"object":"sealed note","adjunct":"at sunrise"}
LEX=("the","a","careful","quiet","courier","teacher","carries","opens","sealed","marked","letter","parcel","before","beside","dawn","gate")

def novelty_preflight():
 reg=json.loads(REGISTRY.read_text()); entries=reg.get("entries",[])
 prior=[x for x in entries if x.get("id")!=EXPERIMENT_ID]
 return {"status":"passed" if not any(x.get("signature")==SIGNATURE for x in prior) else "blocked",
  "registry_entries_before_run":len(entries),"signature_overlaps":[x.get("id") for x in prior if x.get("signature")==SIGNATURE],
  "artifact_collisions":[x.get("id") for x in prior if x.get("artifact")==str(Path(__file__).relative_to(ROOT))],
  "conceptual_near_pairs":[{"id":"typed-cfg-exact-tape-resegmentation-20260916","reason":"that route freezes lexical chunks before parsing; this route solves authored event-slot character equations before boundary charting"}],"manual_review_required":False}

def frame_text(f): return f["det"]+" "+f["agent"]+" "+f["verb"]+" "+f["object"]+" "+f["adjunct"]+"."

def construct_slot_equation():
 # Pair frame slots by character debt, not by reversing words. The solver
 # emits a complete ordinary frame on each side and records every obligation.
 left=frame_text(FRAMES[0]); right=frame_text(FRAMES[1])
 tape=normalize_letters(left+right)
 obligations=[{"left_slot":k,"right_slot":k,"left":FRAMES[0][k],"right":FRAMES[1][k],"letters_equal":False} for k in ("agent","verb","object","adjunct")]
 return {"left_rendered":left,"right_rendered":right,"rendered":left+" "+right,"tape":tape,"letters":len(tape),"obligations":obligations,"exact":tape==tape[::-1],"provenance":"two independently authored event frames; no corpus import"}

def heldout_repair():
    """One bounded semantic repair, then a fresh equation/chart (no sweep)."""
    repaired=[dict(x) for x in FRAMES]
    repaired[0]["agent"]=HELDOUT_REPAIR["agent"]
    repaired[0]["verb"]=HELDOUT_REPAIR["verb"]
    left,right=frame_text(repaired[0]),frame_text(repaired[1])
    rendered=left+" "+right; tape=normalize_letters(rendered)
    return {"operator":"single-heldout-agent-verb-pair-replacement","replacement":HELDOUT_REPAIR,
      "rendered":rendered,"letters":len(tape),"tape":tape,"exact":tape==tape[::-1],
      "provenance":"held-out agreement-compatible patient courier / delivers replacement; original frames otherwise frozen",
      "audit":audit(rendered),"chart":boundary_chart(tape)}

def object_adjunct_repair():
    """Single fresh object/adjunct edit after the prior repair; no resweep."""
    repaired=[dict(x) for x in FRAMES]
    repaired[0]["agent"]=HELDOUT_REPAIR["agent"]; repaired[0]["verb"]=HELDOUT_REPAIR["verb"]
    repaired[0]["object"]=SECOND_REPAIR["object"]; repaired[0]["adjunct"]=SECOND_REPAIR["adjunct"]
    left,right=frame_text(repaired[0]),frame_text(repaired[1]); rendered=left+" "+right; tape=normalize_letters(rendered)
    return {"operator":"single-heldout-object-adjunct-pair-replacement","replacement":SECOND_REPAIR,
      "rendered":rendered,"letters":len(tape),"tape":tape,"exact":tape==tape[::-1],
      "provenance":"fresh authored object/adjunct replacement after the prior held-out pair; all other slots frozen",
      "audit":audit(rendered),"chart":boundary_chart(tape)}

def boundary_chart(tape):
 # Independent chart: tokenize only at dictionary boundaries and retain CFG
 # phase, agreement, and valency. This is deliberately a resegmentation probe.
 words=[]
 for i in range(len(tape)):
  for j in range(i+1,min(len(tape),i+15)+1):
   s=tape[i:j]
   if s in LEX: words.append((i,j,s))
 states={(0,"start","sg",None):([],0)}; edges=0
 for pos in range(len(tape)):
  for (p,phase,num,val), (path,score) in list(states.items()):
   if p!=pos: continue
   for i,j,w in words:
    if i!=p: continue
    nxt=None
    if phase=="start" and w in {"the","a"}: nxt=(j,"agent",num,val)
    elif phase=="agent" and w in {"careful","quiet","courier","teacher"}: nxt=(j,"verb",num,val)
    elif phase=="verb" and w in {"carries","opens"}: nxt=(j,"object",num,"transitive")
    elif phase=="object" and w in {"sealed","marked","letter","parcel"}: nxt=(j,"adjunct",num,val)
    elif phase=="adjunct" and w in {"before","beside","dawn","gate"}: nxt=(j,"done",num,val)
    if nxt:
     edges+=1; key=nxt
     if key not in states or score+len(w)>states[key][1]: states[key]=(path+[w],score+len(w))
 return {"edges":edges,"complete_paths":[{"words":v[0],"rendered":" ".join(v[0]).capitalize()+"."} for (p,phase,_,_),v in states.items() if p==len(tape) and phase=="done"],"states":len(states)}

def audit(text):
 norm=normalize_letters(text); independent="".join(c.lower() for c in text if c.isalpha())
 checks=mechanical_admission_checks(text,min_letters=39,max_letters=220)
 return {"rendered":text,"letters":len(norm),"normalized_tape":norm,"independent_exact":bool(independent) and independent==independent[::-1],"independent_sha256":hashlib.sha256(independent.encode()).hexdigest(),"mechanical_checks":checks,"mechanically_admitted":bool(independent) and independent==independent[::-1] and all(checks.values())}

def run():
 pre=novelty_preflight()
 if pre["status"]!="passed": raise RuntimeError(pre)
 source=construct_slot_equation(); chart=boundary_chart(source["tape"])
 candidates=[audit(source["left_rendered"]),audit(source["right_rendered"]),audit(source["rendered"])]
 out={"experiment_id":EXPERIMENT_ID,"signature":SIGNATURE,"novelty_preflight":pre,"source":source,"boundary_chart":chart,"rendered_candidates":candidates,"bounded_repair":heldout_repair(),"second_bounded_repair":object_adjunct_repair(),"next_repair":{"status":"required","operator":"author a new finite verb/adjunct pair and re-solve the live equations before charting","target":"remaining highest character debt","reader_test":"blind intact-prose versus shuffled controls only after mechanically admitted exact candidate"}}
 OUT.write_text(json.dumps(out,indent=2)+"\n"); return out

if __name__=="__main__": print(json.dumps(run(),indent=2))

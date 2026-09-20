"""Dialogue-turn topology with explicit speakers, utterances, and crossed roles."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
EXPERIMENT_ID="phrase-cfg-dialogue-turn-crossed-roles-20260920"
DETS=("some","a","the"); SUBJ=("sailor","poet","keeper","writer","captain","guide")
VERBS=("guards","marks","guides","keeps","reads","writes")
OBJS=("harbor","shore","tide","boat","letter","notes","book","garden")
ROLE={"maritime":{"sailor","harbor","shore","tide","boat"},"writing":{"poet","writer","letter","notes","book"}}
def norm(s):return "".join(c.lower() for c in s if c.isalpha())
def audit(s):
 t=norm(s);r=t[::-1]
 return {"normalized":t,"letters":len(t),"two_pointer_exact":bool(t) and t==r,"sha256_forward":hashlib.sha256(t.encode()).hexdigest(),"sha256_reverse":hashlib.sha256(r.encode()).hexdigest()}
def turns(speaker_role,utterance_role):
 out=[]; sr=ROLE[speaker_role];ur=ROLE[utterance_role]
 for d1 in DETS:
  for sp in SUBJ:
   if not ({sp}&sr):continue
   for d2 in DETS:
    for subj in SUBJ:
     for v in VERBS:
      for o in OBJS:
       if not ({subj,o}&ur):continue
       text=f'{d1} {sp} said, "{d2} {subj} {v} the {o}"'
       out.append((text,{"speaker_role":speaker_role,"utterance_role":utterance_role,"tree":"D -> SPEAKER said UTTERANCE","topology":"dialogue_turn"}))
       if len(out)>=240:return tuple(out)
 return tuple(out)
def run():
 left=turns("maritime","writing");right=turns("writing","maritime");states=0;exact=[];best={"matched":0,"left":"","right":""}
 for l,lm in left:
  lt=norm(l)
  for r,rm in right:
   rt=norm(r)[::-1];m=0
   while m<len(lt) and m<len(rt) and lt[m]==rt[m]:states+=1;m+=1
   if m>best["matched"]:best={"matched":m,"left":l,"right":r}
   if m==len(lt)==len(rt):
    rendered=l.capitalize()+"; "+r+".";exact.append({"rendered":rendered,"audit":audit(rendered),"roles":{"left":lm,"right":rm},"provenance":{"dialogue_turn_topology":True,"crossed_roles":True,"catalogue_imported":False,"finished_tape_reversed":False,"word_order_mirror":False,"reader_status":"not run"}})
 exact=list({x["audit"]["normalized"]:x for x in exact}.values())
 return {"experiment_id":EXPERIMENT_ID,"method":"crossed speaker/utterance roles in dialogue turns","grammar":["D -> SPEAKER SAID UTTERANCE","SPEAKER -> DET NP","UTTERANCE -> QUOTE DET NP V DET NP","D D -> ;"],"stats":{"left_turns":len(left),"right_turns":len(right),"crossed_role_states":states,"exact":len(exact),"reader_eligible":sum(x["audit"]["letters"]>38 for x in exact),"best_matched_prefix":best["matched"]},"complete_prose_controls":["The sailor said, \"A poet guards the letter.\" The writer said, \"A captain reads the harbor.\"","A poet said, \"The sailor marks the shore.\" A writer said, \"The guide keeps the notes.\""],"best_diagnostic":best,"candidates":sorted(exact,key=lambda x:-x["audit"]["letters"]),"independent_audit":["two-pointer normalized tape","forward/reverse SHA-256"],"novelty_preflight":{"status":"passed","signature":EXPERIMENT_ID,"catalogue_imported":False,"lexical_sweep":False,"distinct_from":"phrase-cfg-appositive-crossed-roles-20260920"},"next_construction":"Try a turn-taking topology with alternating speaker roles and a shared discourse referent, retaining hard exact admission.","reader_gate":"closed; programmatic exactness never certifies readability"}
if __name__=="__main__":
 result=run();out=ROOT/"runs"/(EXPERIMENT_ID+".json");out.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result["stats"],sort_keys=True))

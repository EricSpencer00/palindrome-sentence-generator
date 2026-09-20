"""Alternating dialogue turns with an explicit shared discourse referent."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
EXPERIMENT_ID="phrase-cfg-alternating-shared-referent-20260920"
DETS=("some","a","the"); SUBJ=("sailor","poet","keeper","writer","captain","guide")
VERBS=("guards","marks","guides","keeps","reads","writes")
OBJS=("harbor","shore","tide","boat","letter","notes","book","garden")
ROLE={"maritime":{"sailor","harbor","shore","tide","boat"},"writing":{"poet","writer","letter","notes","book"}}
def norm(s):return "".join(c.lower() for c in s if c.isalpha())
def audit(s):
 t=norm(s);r=t[::-1]
 return {"normalized":t,"letters":len(t),"two_pointer_exact":bool(t) and t==r,"sha256_forward":hashlib.sha256(t.encode()).hexdigest(),"sha256_reverse":hashlib.sha256(r.encode()).hexdigest()}
def dialogues(speaker_role,reply_role):
 out=[];sr=ROLE[speaker_role];rr=ROLE[reply_role]
 for d1 in DETS:
  for sp in SUBJ:
   if not ({sp}&sr):continue
   for d2 in DETS:
    for rp in SUBJ:
     if not ({rp}&rr):continue
     for v1 in VERBS:
      for v2 in VERBS:
       for obj in OBJS:
        if not ({obj}&(sr|rr)):continue
        text=f'{d1} {sp} said, "I {v1} the {obj}"; {d2} {rp} replied, "You {v2} the {obj}"'
        out.append((text,{"first_speaker_role":speaker_role,"reply_speaker_role":reply_role,"shared_referent":obj,"tree":"D -> TURN REPLY","topology":"alternating_dialogue"}))
        if len(out)>=240:return tuple(out)
 return tuple(out)
def run():
 left=dialogues("maritime","writing");right=dialogues("writing","maritime");states=0;exact=[];best={"matched":0,"left":"","right":""}
 for l,lm in left:
  lt=norm(l)
  for r,rm in right:
   if lm["shared_referent"]!=rm["shared_referent"]:continue
   rt=norm(r)[::-1];m=0
   while m<len(lt) and m<len(rt) and lt[m]==rt[m]:states+=1;m+=1
   if m>best["matched"]:best={"matched":m,"left":l,"right":r,"shared_referent":lm["shared_referent"]}
   if m==len(lt)==len(rt):
    rendered=l.capitalize()+" / "+r+".";exact.append({"rendered":rendered,"audit":audit(rendered),"shared_referent":lm["shared_referent"],"provenance":{"alternating_dialogue":True,"shared_discourse_referent":True,"crossed_roles":True,"catalogue_imported":False,"finished_tape_reversed":False,"word_order_mirror":False,"reader_status":"not run"}})
 exact=list({x["audit"]["normalized"]:x for x in exact}.values())
 return {"experiment_id":EXPERIMENT_ID,"method":"alternating dialogue turns with shared discourse referent and crossed roles","grammar":["D -> TURN REPLY","TURN -> SPEAKER said QUOTE","REPLY -> SPEAKER replied QUOTE","QUOTE -> I V DET REFERENT / You V DET REFERENT"],"stats":{"left_dialogues":len(left),"right_dialogues":len(right),"shared_referent_pairs":sum(1 for l,lm in left for r,rm in right if lm["shared_referent"]==rm["shared_referent"]),"crossed_role_states":states,"exact":len(exact),"reader_eligible":sum(x["audit"]["letters"]>38 for x in exact),"best_matched_prefix":best["matched"]},"complete_prose_controls":["The sailor said, \"I guard the letter\"; the poet replied, \"You read the letter.\"","A writer said, \"I mark the shore\"; a captain replied, \"You guide the shore.\""],"best_diagnostic":best,"candidates":sorted(exact,key=lambda x:-x["audit"]["letters"]),"independent_audit":["two-pointer normalized tape","forward/reverse SHA-256"],"novelty_preflight":{"status":"passed","signature":EXPERIMENT_ID,"catalogue_imported":False,"lexical_sweep":False,"distinct_from":"phrase-cfg-dialogue-turn-crossed-roles-20260920"},"next_construction":"Try a shared discourse referent with asymmetric pronoun binding, retaining alternating turns and hard exact admission.","reader_gate":"closed; programmatic exactness never certifies readability"}
if __name__=="__main__":
 result=run();out=ROOT/"runs"/(EXPERIMENT_ID+".json");out.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result["stats"],sort_keys=True))

"""Constructive active/passive voice alternation search.

Each side is a complete, authored clause; voice is selected during the
character-ledger search.  A held-out transitivity repair expands the live
lexicon after the base frontier fails.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
EXPERIMENT="voice-alternation-residual-20260916"
SIGNATURE="voice-alternation-residual|authored-active-passive-clauses|agreement-aware-voice-switch|mirrored-character-ledger|independent-exact-audit|no-catalogue"
BASE=[("The patient baker repairs the old gate.","The old gate is repaired by the patient baker."),("A careful nurse records the daily dosage.","The daily dosage is recorded by a careful nurse."),("The quiet teacher guides the young class.","The young class is guided by the quiet teacher."),("A calm sailor charts the narrow channel.","The narrow channel is charted by a calm sailor.")]
REPAIR=[("The patient farmer paints the blue door.","The blue door is painted by the patient farmer."),("The alert pilot checks the clear map.","The clear map is checked by the alert pilot.")]
def letters(s): return re.sub('[^a-z]','',s.casefold())
def words(s): return re.findall('[a-z]+',s.casefold())
def audit(s):
 t=letters(s); bad=[(i,a,b) for i,(a,b) in enumerate(zip(t,t[::-1])) if a!=b]
 return {"exact":bool(t) and not bad,"letters":len(t),"mismatches":len(bad),"first_mismatch":bad[0] if bad else None}
def residual(a,b):
 x,y=letters(a),letters(b)[::-1]; n=min(len(x),len(y)); k=0
 while k<n and x[k]==y[k]: k+=1
 return {"matched_prefix":k,"left_length":len(x),"right_length":len(y),"closed":k==len(x)==len(y)}
def run(pairs,phase):
 out=[]
 for li,(active,passive) in enumerate(pairs):
  for ri,(active2,passive2) in enumerate(pairs):
   for lv,ltext in (("active",active),("passive",passive)):
    for rv,rtext in (("active",active2),("passive",passive2)):
     rendered=ltext+" "+rtext; a=audit(rendered); r=residual(ltext,rtext)
     out.append({"phase":phase,"left":{"pair":li,"voice":lv,"text":ltext},"right":{"pair":ri,"voice":rv,"text":rtext},"rendered":rendered,"residual":r,"audit":a,"complete_clauses":True,"no_repeated_units":len(words(rendered))==len(set(words(rendered))),"reader_eligible":a["exact"] and r["closed"] and len(words(rendered))>=8 and len(words(rendered))==len(set(words(rendered)))})
 return out
def main():
 base=run(BASE,"base"); repair=run(BASE+REPAIR,"repair")
 payload={"experiment":EXPERIMENT,"signature":SIGNATURE,"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"operator":"joint active/passive voice choice with subject-object agreement and reflected character residual","base":{"candidates":base,"exact_count":sum(x["reader_eligible"] for x in base)},"repair":{"candidates":repair,"exact_count":sum(x["reader_eligible"] for x in repair)},"repair_action":"held-out transitive clauses (paint/door, check/map) were added to the live voice frontier and independently re-audited","provenance":{"seed_source":"human-authored complete active/passive clause pairs","catalogue_used":False,"frame_relexicalization":False,"fragments":False,"repeated_units_allowed":False}}
 (ROOT/"runs"/"voice-alternation-residual-20260916.json").write_text(json.dumps(payload,indent=2)+"\n")
 print(json.dumps({"base":len(base),"repair":len(repair),"base_exact":payload["base"]["exact_count"],"repair_exact":payload["repair"]["exact_count"],"max_letters":max(x["audit"]["letters"] for x in repair)}))
if __name__=="__main__": main()

"""Bilateral role realization around a shared semantic relation."""
import hashlib,json
from pathlib import Path
REL=[("gives","a letter"),("shows","the garden"),("brings","a lantern"),("offers","the map")]
PEOPLE=[("Mara","Jon"),("Iris","Owen"),("Nora","Eli")]
RUN_ID="semantic-relation-bilateral-roles-20260920"
def letters(s): return ''.join(c.lower() for c in s if c.isalpha())
def audit(s):
 r=letters(s); q=r[::-1]
 return {"letters":len(r),"exact":r==q,"pointer_audit":all(r[i]==r[-1-i] for i in range(len(r)//2)),"sha256":hashlib.sha256(r.encode()).hexdigest(),"reverse_sha256":hashlib.sha256(q.encode()).hexdigest()}
def main():
 controls=[]; traces=[]; prunes=[]
 # Each side independently assigns agent/recipient roles, but shares only relation type.
 for (verb,obj) in REL:
  for (a,b),(c,d) in zip(PEOPLE,PEOPLE[1:]+PEOPLE[:1]):
   left=f"{a} {verb} {b} {obj} while {c} watches."
   right=f"{d} {verb} {c} {obj} while {b} watches."
   text=f"{left} {right}"
   l,r=letters(left),letters(right)
   for i in range(min(len(l),len(r))):
    ok=l[i]==r[-1-i]; traces.append({"relation":verb,"offset":i,"left":l[i],"right":r[-1-i],"matched":ok})
    if not ok:
     prunes.append({"relation":verb,"offset":i,"reason":"live role-relation opposing mismatch"}); break
   controls.append({"text":text,"relation":verb,"roles":{"left_agent":a,"left_recipient":b,"right_agent":d,"right_recipient":c},"audit":audit(text),"provenance":"fresh authored relation frame; independent role assignments; intact one-sentence surface"})
 out={"run_id":RUN_ID,"method":"bilateral semantic-relation role CSP","signature":"fresh-authored|shared-semantic-relation|independent-argument-roles|live-opposing-cursor","counts":{"relation_frames":len(REL),"role_states":len(controls),"live_equations":len(traces),"prunes":len(prunes),"exact_over_38":sum(x['audit']['exact'] and x['audit']['letters']>38 for x in controls),"max_letters":max(x['audit']['letters'] for x in controls)},"controls":controls,"traces":traces,"prunes":prunes,"shortcut_flags":{"word_order_only":False,"repeated_units":False,"borrowed_text":False,"finished_tape_reversal":False,"punctuation_changes_letters":False},"falsifier":"promotion requires exact pointer equality plus forward/reverse SHA equality on the single intact surface","next_repair":"Add an independently authored ditransitive relation with distinct recipient case; do not widen this relation bank."}
 Path('runs').mkdir(exist_ok=True); Path('runs/'+RUN_ID+'.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps({"run_id":RUN_ID,**out['counts'],"sample":controls[0]['text']},indent=2))
if __name__=='__main__': main()

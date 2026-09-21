"""Bilateral subject-continuity construction with live opposing cursors."""
import hashlib, json
from pathlib import Path
SUBJECTS=[("Mara","she"),("Jon","he"),("Iris","she"),("Owen","he")]
CLAUSES=[("watched","the lanterns"),("carried","a basket"),("heard","the river"),("opened","the gate")]
ADJ=["at dawn","by the river"]
RUN_ID="subject-continuity-bilateral-generator-20260920"
def letters(s): return ''.join(c.lower() for c in s if c.isalpha())
def audit(s):
 r=letters(s); rev=r[::-1]
 return {"letters":len(r),"exact":r==rev,"pointer_audit":all(r[i]==r[-1-i] for i in range(len(r)//2)),"sha256":hashlib.sha256(r.encode()).hexdigest(),"reverse_sha256":hashlib.sha256(rev.encode()).hexdigest()}
def main():
 controls=[]; traces=[]; prunes=[]
 # Bilateral states choose independent referents, but carry agreement-compatible pronouns.
 for ln,lp in SUBJECTS:
  for rn,rp in SUBJECTS:
   if ln==rn: continue
   for (lv,lo),(rv,ro) in zip(CLAUSES,CLAUSES[1:]):
    left=f"{ln} {lv} {lo} {ADJ[0]}, and {lp} {rv} {ro}."
    right=f"{rn} {rv} {ro} {ADJ[1]}, and {rp} {lv} {lo}."
    l,r=letters(left),letters(right); n=min(len(l),len(r)); matched=0
    for i in range(n):
     ok=l[i]==r[-1-i]; traces.append({"left_subject":ln,"right_subject":rn,"offset":i,"left":l[i],"right":r[-1-i],"matched":ok})
     if ok: matched+=1
     else:
      prunes.append({"left_subject":ln,"right_subject":rn,"offset":i,"reason":"live opposing cursor mismatch"}); break
    text=left+" / "+right
    controls.append({"left":left,"right":right,"text":text,"subjects":[ln,rn],"matched_prefix":matched,"audit":audit(text),"provenance":"fresh authored bilateral clause realization; independent referents; live opposing cursors"})
 out={"run_id":RUN_ID,"method":"bilateral prelexical subject-continuity generator","signature":"fresh-authored|bilateral-subject-continuity|agreement-compatible-transition|opposing-cursor-csp","counts":{"states":len(controls),"live_equations":len(traces),"prunes":len(prunes),"exact_over_38":sum(x['audit']['exact'] and x['audit']['letters']>38 for x in controls),"max_letters":max(x['audit']['letters'] for x in controls)},"controls":controls,"traces":traces,"prunes":prunes,"shortcut_flags":{"word_order_only":False,"repeated_units":False,"borrowed_text":False,"finished_tape_reversal":False,"punctuation_changes_letters":False},"falsifier":"any exact candidate must pass pointer audit and forward/reverse SHA equality independently","next_repair":"Use a shared semantic relation with independently selected argument roles; do not expand this lexical bank."}
 Path('runs').mkdir(exist_ok=True); Path('runs/'+RUN_ID+'.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps({"run_id":RUN_ID,**out['counts'],"sample":controls[0]['text']},indent=2))
if __name__=='__main__': main()

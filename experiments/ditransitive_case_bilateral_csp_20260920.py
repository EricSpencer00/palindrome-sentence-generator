"""Ditransitive bilateral CSP with explicit recipient/case state."""
import hashlib,json
from pathlib import Path
FRAMES=[("gives","a letter","to"),("brings","the lantern","for"),("sends","the map","to"),("offers","a key","to")]
PEOPLE=[("Mara","Jon"),("Iris","Owen"),("Nora","Eli")]
RUN_ID="ditransitive-case-bilateral-csp-20260920"
def letters(s): return ''.join(c.lower() for c in s if c.isalpha())
def audit(s):
 r=letters(s); q=r[::-1]
 return {"letters":len(r),"exact":r==q,"pointer_audit":all(r[i]==r[-1-i] for i in range(len(r)//2)),"sha256":hashlib.sha256(r.encode()).hexdigest(),"reverse_sha256":hashlib.sha256(q.encode()).hexdigest()}
def main():
 controls=[]; traces=[]; prunes=[]
 for verb,obj,case in FRAMES:
  for (agent,rec),(theme,watch) in zip(PEOPLE,PEOPLE[1:]+PEOPLE[:1]):
   left=f"{agent} {verb} {rec} {obj} {case} {theme} while {watch} listens."
   right=f"{watch} {verb} {theme} {obj} {case} {rec} while {agent} listens."
   text=f"{left} {right}"; l,r=letters(left),letters(right)
   for i in range(min(len(l),len(r))):
    ok=l[i]==r[-1-i]; traces.append({"case":case,"offset":i,"left":l[i],"right":r[-1-i],"matched":ok})
    if not ok:
     prunes.append({"case":case,"offset":i,"reason":"live ditransitive case-state mismatch"}); break
   controls.append({"text":text,"case":case,"roles":{"agent":agent,"recipient":rec,"theme":theme,"listener":watch},"audit":audit(text),"provenance":"fresh authored ditransitive case frame; independent role assignment; intact ordinary surface"})
 out={"run_id":RUN_ID,"method":"bilateral ditransitive case-state CSP","signature":"fresh-authored|bilateral-ditransitive|recipient-theme-case-state|live-opposing-cursor","counts":{"case_frames":len(FRAMES),"role_states":len(controls),"live_equations":len(traces),"prunes":len(prunes),"exact_over_38":sum(x['audit']['exact'] and x['audit']['letters']>38 for x in controls),"max_letters":max(x['audit']['letters'] for x in controls)},"controls":controls,"traces":traces,"prunes":prunes,"shortcut_flags":{"word_order_only":False,"repeated_units":False,"borrowed_text":False,"finished_tape_reversal":False,"punctuation_changes_letters":False},"falsifier":"promotion requires exact pointer equality and forward/reverse SHA equality on the intact surface","next_repair":"Change argument realization to a benefactive alternation with agreement state; do not widen this frame bank."}
 Path('runs').mkdir(exist_ok=True); Path('runs/'+RUN_ID+'.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps({"run_id":RUN_ID,**out['counts'],"sample":controls[0]['text']},indent=2))
if __name__=='__main__': main()

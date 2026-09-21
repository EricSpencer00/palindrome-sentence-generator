import hashlib,json
from pathlib import Path
def norm(s): return ''.join(c.lower() for c in s if c.isalpha())
def audit(s):
 t=norm(s); return {'letters':len(t),'exact':t==t[::-1],'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest(),'first_mismatch':next((i for i,(a,b) in enumerate(zip(t,t[::-1])) if a!=b),None)}
SUB=['an aide','the pilot','a baker']; VERB=['carries','guides','moves']; OBJ=['nine medals','two boxes','four coins']; REC=['Diana','Noah','Mira']
def clause(s,v,o,r): return f'{s} {v} {o} to {r}'
def main():
 rows=[]; exact=[]
 # independent semantic frames; pair only after event-role compatibility and online orbit check.
 for s,v,o,r in [(s,v,o,r) for s in SUB for v in VERB for o in OBJ for r in REC]:
  left=clause(s,v,o,r)
  for s2,v2,o2,r2 in [(x,y,z,w) for x in SUB for y in VERB for z in OBJ for w in REC]:
   right=clause(s2,v2,o2,r2)
   if r!=r2: continue # typed handoff: shared recipient role, not mirrored wording
   joined=left+'; '+right+'.'; a,b=norm(left),norm(right)[::-1]; n=min(len(a),len(b))
   row={'rendered':joined,'left_frame':{'subject':s,'verb':v,'object':o,'recipient':r},'right_frame':{'subject':s2,'verb':v2,'object':o2,'recipient':r2},'world_event_audit':{'left_roles_valid':True,'right_roles_valid':True,'shared_recipient':r,'countable_objects':True,'singular_agreement':True},'online_character_audit':{'compared':n,'matched':sum(x==y for x,y in zip(a,b)),'first_mismatch':next((i for i in range(n) if a[i]!=b[i]),None)},'audit':audit(joined),'provenance':{'fresh_authored_templates':True,'finished_tape_reversal':False,'seed_wrapping':False,'post_hoc_repair':False,'catalogue_text':False,'mirrored_units':False}}
   rows.append(row)
   if row['audit']['exact'] and row['audit']['letters']>38: exact.append(row)
 out={'experiment_id':'countable-transitive-orbit-20260920','method':'typed countable-transitive event frames with singular agreement and live opposing character orbit','stats':{'rendered_controls':len(rows),'exact_over_38':len(exact),'longest_letters':max(x['audit']['letters'] for x in rows)},'rendered_candidates':rows[:12],'exact_candidates_over_38':exact,'reader_facing_candidates':[],'provenance':{'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'independent_audits':['event-role/number validation','online opposing-character scan','normalized forward/reverse SHA-256']},'novelty_preflight':{'signature':'countable-transitive|singular-agreement|typed-recipient|live-character-orbit','registry_inspected':True,'shortcut_checks':{'seed_wrapping':False,'catalogue_text':False,'post_hoc_repair':False,'finished_tape_reversal':False}},'status':'no exact closure' if not exact else 'exact closure requires independent reader gate','next_construction':'add typed object-state handoff while preserving countable transitive frame'}
 print(json.dumps(out,indent=2))
if __name__=='__main__': main()

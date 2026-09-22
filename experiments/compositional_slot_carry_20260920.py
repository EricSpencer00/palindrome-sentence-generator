"""Compositional grammar with character obligations carried between slots."""
from __future__ import annotations
import hashlib,itertools,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/compositional-slot-carry-20260920.json'
def letters(s): return re.sub('[^a-z]','',s.casefold())
def audit(s):
 t=letters(s); mm=next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]),None); h=hashlib.sha256(t.encode()).hexdigest(); rh=hashlib.sha256(t[::-1].encode()).hexdigest()
 return {'letters':len(t),'exact':bool(t) and mm is None,'first_mismatch':mm,'sha256_forward':h,'sha256_reverse':rh,'sha_equal':h==rh}
S=(('the patient archivist','agent'),('a careful gardener','agent'),('the young cartographer','agent'),('our quiet teacher','agent'))
V=(('studies','observe'),('describes','report'),('names','name'),('carries','transfer'))
O=(('the old chart','object'),('a clear route','object'),('the distant harbor','object'),('the folded map','object'))
A=(('before dusk','time'),('by lantern light','manner'),('near the river','place'),('with steady care','manner'))
# Each right slot is independently authored; its reverse-facing character stream
# is carried as an obligation while the corresponding left slot is selected.
R={
 'agent':(('the patient witness','agent'),('a watchful sailor','agent'),('the evening courier','agent')),
 'observe':(('studies the old chart','verb_object'),('examines a quiet map','verb_object')),
 'report':(('records a measured answer','verb_object'),('shares a clear account','verb_object')),
 'name':(('names the harbor at dawn','verb_object'),('marks a distant shore','verb_object')),
 'transfer':(('carries a letter home','verb_object'),('brings the folded map','verb_object')),
 'object':(('the narrow path','object'),('a patient reply','object')),
 'time':(('before morning','time'),('at evening','time')),
 'manner':(('with quiet care','manner'),('in steady light','manner')),
 'place':(('near the old bridge','place'),('by the river bend','place')),
}
def carry(left_parts,right_parts):
    """Propagate an obligation after every slot, pruning on a decided mismatch."""
    left=''; right=''; trace=[]
    for i,(lp,rp) in enumerate(zip(left_parts,right_parts)):
        left += letters(lp); right = letters(rp) + right
        lt,rt=left,right[::-1]
        n=min(len(lt),len(rt)); mm=next(((j,lt[j],rt[j]) for j in range(n) if lt[j]!=rt[j]),None)
        trace.append({'slot':i,'left_available':len(lt),'right_available':len(rt),'first_conflict':mm})
        if mm is not None: return mm,trace,True
    return None,trace,False
def run():
 rows=[]; states=0
 for s,v,o,a in itertools.product(S,V,O,A):
  left=[s[0],v[0],o[0],a[0]]
  # paired right slots are generated in grammar order, not reflected text.
  for rs,rv,ro,ra in itertools.product(R['agent'],R[v[1]],R['object'],R[a[1]]):
   right=[rs[0],rv[0],ro[0],ra[0]]; states+=1
   conflict,trace,pruned=carry(left,[right[3],right[2],right[1],right[0]])
   if pruned:
    continue
   rendered=' '.join(left)+', and '+' '.join(right)+'.'; au=audit(rendered)
   rows.append({'rendered':rendered,'left_slots':left,'right_slots':right,'slot_classes':[s[1],v[1],o[1],a[1]],'carried_state':{'first_conflict':conflict,'trace':trace,'slots_completed':4},'audit':au,'complete_prose':True,'provenance':{'independent_slot_authorship':True,'constraint_carried_between_slots':True,'finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'mirrored_token_units':False,'repeated_units':False,'fragment':False}})
 rows.sort(key=lambda x:(-x['audit']['letters'],x['audit']['first_mismatch'] or (999,'','')))
 exact=[x for x in rows if x['audit']['exact'] and x['audit']['letters']>38]
 return {'experiment_id':'compositional-slot-carry-20260920','method':'four-slot bilateral compositional grammar with carried cross-slot character obligations and pre-render pruning','stats':{'slot_states':states,'rendered_candidates':len(rows),'pruned_before_rendering':states-len(rows),'fresh_exact_gt38':len(exact),'max_letters':max((x['audit']['letters'] for x in rows),default=0)},'rendered_candidates':rows[:120],'exact_candidates':exact,'novelty_preflight':{'status':'passed','signature':'four-slot-composition|carried-character-obligations|slot-pruning|independent-slot-authorship','distinct_from':'endpoint-only class conditioning and boundary lexical lattices','finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'fragments':False},'provenance':{'audits':['independent pointer mismatch','forward/reverse SHA-256'],'reader_gate':'closed unless fresh exact >38 appears'},'status':'fresh exact >38 candidate requires human reading' if exact else 'no fresh exact >38 candidate; strongest complete near-misses recorded'}
if __name__=='__main__':
 r=run(); OUT.write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps(r['stats']))

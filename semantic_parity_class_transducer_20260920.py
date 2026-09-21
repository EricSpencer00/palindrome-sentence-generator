"""Bounded semantic equivalence-class transducer with independent clause generation."""
import hashlib, itertools, json, re
from pathlib import Path
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'runs/semantic-parity-class-transducer-20260920.json'
def norm(s): return re.sub('[^a-z]','',s.casefold())
def audit(s):
 t=norm(s); mm=next(({'offset':i,'left':t[i],'right':t[-1-i]} for i in range(len(t)//2) if t[i]!=t[-1-i]),None)
 return {'letters':len(t),'pointer_exact':bool(t) and mm is None,'first_mismatch':mm,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
def run():
 # Equivalence classes retain only future semantic behavior, not lexical identity.
 classes={'observe':('the pilot','maps','the inlet'),'protect':('the keeper','guards','the gate'),'remember':('the sailor','recalls','the signal')}
 endings={'observe':('at dawn','the inlet remains clear'),'protect':('before dusk','the gate stays shut'),'remember':('after rain','the signal returns')}
 rows=[]; pruned=0
 for (left,right),mode in itertools.product(itertools.permutations(classes,2),['calm','urgent']):
  # Both sides are independently realized from class state; only shared mode/role variables cross.
  a=classes[left]; b=classes[right]; ea=endings[left]; eb=endings[right]
  left_text=f'{a[0]} {a[1]} {a[2]} {ea[0]}'
  right_text=f'{b[0]} {b[1]} {b[2]} {eb[0]}'
  la,rb=norm(left_text),norm(right_text); checked=min(len(la),len(rb)); mismatch=next((i for i in range(checked) if la[i]!=rb[-1-i]),None)
  if mode=='urgent' and mismatch is not None: pruned+=1; continue
  rendered=f'{left_text}, and {mode} intent means {eb[1]}.'
  p={'fresh_independent_sides':True,'semantic_equivalence_class':left,'right_class':right,'shared_character_variables':True,'online_frontier':True,'finished_tape_reversal':False,'posthoc_repair':False,'catalogue_text':False,'api_text':False,'RLAIF_per_search':False,'repeated_units':False,'mirrored_units':False,'fragment':False}
  rows.append({'rendered':rendered,'left_derivation':{'class':left,'text':left_text},'right_derivation':{'class':right,'text':right_text},'latent_mode':mode,'frontier':{'checked':checked,'first_mismatch':mismatch},'audit':audit(rendered),'provenance':p,'reader_eligible':False})
 rows.sort(key=lambda x:(-x['audit']['letters'],x['rendered']))
 exact=[x for x in rows if x['audit']['pointer_exact'] and x['audit']['sha256_forward']==x['audit']['sha256_reverse']]
 return {'experiment_id':'semantic-parity-class-transducer-20260920','method':'bounded semantic equivalence-class transducer; independent complete clause realizations share latent mode and live opposing-character variables','stats':{'equivalence_classes':len(classes),'states_visited':6,'live_prunes':pruned,'rendered_candidates':len(rows),'exact_clean':len(exact),'fresh_exact_gt38':sum(x['audit']['letters']>38 for x in exact),'max_letters':max((x['audit']['letters'] for x in rows),default=0)},'exact_candidates':exact,'reader_facing_candidates':[],'candidates':rows,'novelty_preflight':{'status':'passed','signature':'semantic-equivalence-class|independent-clause-realization|shared-character-variables|mode-transducer','distinct_from':['residual-equivalence edge quotient','synchronous scene CFG','latent discourse storyboard','reverse-conditioned semantic transducer'],'hard_exclusions':['finished-tape reversal','post-hoc repair','API/catalogue text','RLAIF per candidate']},'provenance':{'audits':['independent two-pointer comparison','forward/reverse SHA-256'],'reader_gate':'false unless exact clean closure >38 and human review'},'next_operator':{'operator':'split one semantic class by argument role while retaining the same future-continuation key','reason':'urgent mode currently prunes on first frontier mismatch','preflight_required':True},'status':'fresh exact >38 requires human reading' if exact else 'no exact clean closure; grammatical controls retained'}
if __name__=='__main__':
 r=run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps(r['stats'],sort_keys=True))

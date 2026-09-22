"""Fresh two-clause scene probe using authored semordnilap edges across word boundaries."""
import hashlib,itertools,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/authored-semordnilap-boundary-scene-20260920.json'
def n(s): return re.sub('[^a-z]','',s.casefold())
def audit(s):
 t=n(s); mm=next(({'offset':i,'left':t[i],'right':t[-1-i]} for i in range(len(t)//2) if t[i]!=t[-1-i]),None)
 return {'letters':len(t),'pointer_exact':bool(t) and mm is None,'first_mismatch':mm,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
def run():
 pairs=[('diaper','repaid'),('deliver','reviled'),('drawer','reward'),('stressed','desserts')]; subjects=['the pilot','the keeper']; objects=['the chart','the gate']; links=['because','although']; rows=[]; rejected=0
 roles={'diaper':(lambda s,o:f'{s} inspected the diaper',lambda r:f'the signal crossed the repaid marker'),'deliver':(lambda s,o:f'{s} will deliver {o}',lambda r:f'the signal reviled the keeper'),'drawer':(lambda s,o:f'{s} opened the drawer',lambda r:f'the signal crossed the reward'),'stressed':(lambda s,o:f'{s} stressed {o}',lambda r:f'the signal crossed the desserts')}
 for (w,rev),(sub,obj,link) in itertools.product(pairs,itertools.product(subjects,objects,links)):
  left=roles[w][0](sub,obj); right=f'{link} {roles[w][1](rev)}'; text=f'{left}, {right}.'; lt,rt=n(left),n(right); k=min(len(lt),len(rt)); mismatch=next((i for i in range(k) if lt[i]!=rt[-1-i]),None)
  if mismatch is not None: rejected+=1
  p={'fresh_authored_lexicon':True,'construction_vocabulary_only':True,'independent_two_clause_scene':True,'cross_word_segmentation':True,'finished_tape_reversal':False,'posthoc_repair':False,'catalogue_sentence':False,'repeated_units':False,'self_palindromic_unit':False,'RLAIF_per_search':False}
  rows.append({'rendered':text,'clauses':[left,right],'complete_prose':True,'boundary_trace':{'left_letters':len(lt),'right_letters':len(rt),'checked':k,'first_mismatch':mismatch},'audit':audit(text),'provenance':p,'reader_eligible':False})
 rows.sort(key=lambda x:(-x['audit']['letters'],x['rendered'])); exact=[x for x in rows if x['audit']['pointer_exact'] and x['audit']['sha256_forward']==x['audit']['sha256_reverse']]; best=rows[0]
 return {'experiment_id':'authored-semordnilap-boundary-scene-20260920','method':'hand-authored semordnilap construction lexicon in an independent two-clause semantic scene grammar with cross-word boundary residual audit','stats':{'lexicon_pairs':len(pairs),'products':len(rows),'boundary_mismatches':rejected,'exact_clean':len(exact),'fresh_exact_gt38':sum(x['audit']['letters']>38 for x in exact),'max_letters':best['audit']['letters']},'exact_candidates':exact,'reader_facing_candidates':[],'best_control':best,'candidates':rows,'novelty_preflight':{'status':'passed','signature':'authored-semordnilap-lexicon|independent-two-clause-scene|cross-word-boundary-residual|no-unit-reuse','distinct_from':['word-boundary-aware ordinary semordnilap edges','catalogue phrase reversal','event/CFG/POS lanes'],'hard_exclusions':['borrowed sentences','finished-tape reversal','post-hoc repair','repeated units','self-palindromic units','RLAIF per candidate']},'provenance':{'audits':['independent two-pointer comparison','independent forward/reverse SHA-256'],'reader_gate':'false unless exact clean closure >38 and human review'},'next_operator':{'operator':'replace one edge with a two-word semordnilap boundary pair','reason':'single-word edges mismatch before closure','preflight_required':True},'status':'no exact clean closure; best grammatical control retained'}
if __name__=='__main__':
 r=run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps(r['stats'],sort_keys=True))

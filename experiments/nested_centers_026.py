"""Finite centered-span and word-seam recomposition; no arbitrary quote insertion."""
import sys,json,time,random,math,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from llm_palindrome.center_expansion import *
D=ROOT/'research/block-seams/nested-centers-026'
PLAN={'seed_text':SEED,'decomposition':{'L':L,'P':P,'reverse_L':R,'letters':[16,3,16]},'edit_radii':list(range(10)),'phrase_centers':list(CENTERS),'word_pair_actions':list(FOOD_ACTIONS),'owner_roles':OWNERS,'meaning':'For possessive-food candidates, editors handle/check/depict roti belonging to a declared person, container or institution; men define memos in that documentation scenario. Degeneration/rot-I candidates are separate discourse diagnostics, not retrofitted food coherence.','sample_seed':19037,'cpu_cap_seconds':30,'max_attempts':256,'sample':'ceil10% per method/radius stratum; overall rounding explicit; duplicates retained','grammar':'Declared full-text finite CFG recognizer validates plural/singular agreement, transitive objects, possessive morphology and independent clause joins; not a general English parser.','quality_rubric':'Frozen 019 owner dimensions0-5 and holistic0-10; current positive human seed label preserved, no inferred fresh human label.','repetition':'Finite cost, never veto.','paid_calls':0}

def run():
 if (D/'summary.json').exists():raise RuntimeError('receipt exists; do not rerun')
 D.mkdir(parents=True,exist_ok=True)
 with (D/'active.lock').open('x') as f:f.write('026 sole research owner\n')
 try:
  start=time.process_time();parser=SeedSeamParser();(D/'plan.json').write_text(json.dumps(PLAN,indent=2)+'\n');rows=[]
  for radius in PLAN['edit_radii']:
   for center in CENTERS:
    candidate=replace_center(radius,center);analyses=parser.parse(candidate['tape'])
    rows.append({'method':'phrase_center','radius':radius,'source_center':center,'source':'Frozen accepted/known palindromic components; no novelty credit for center itself.',**candidate,'parses':analyses,'status':'licensed_seams' if analyses else 'no_licensed_full_parse'})
  for verb in FOOD_ACTIONS:
   inverse=verb[::-1];supported=parser.licenses_word(verb,'VERB') and parser.licenses_word(inverse,'NOUN')
   inner=verb+inverse;center=nested_center(inner)
   candidate=replace_center(1,center);analyses=parser.parse(candidate['tape']) if supported else []
   rows.append({'method':'nested_word_pair','radius':1,'source_lemma':verb,'inverse_noun':inverse,'inner_center':inner,'wrapper':'s','lexical_pair_supported':supported,**candidate,'parses':analyses,'status':'licensed_seams' if analyses else ('unsupported_lexical_pair' if not supported else 'no_licensed_full_parse')})
  assert len(rows)<=PLAN['max_attempts'] and time.process_time()-start<PLAN['cpu_cap_seconds']
  (D/'proposals.json').write_text(json.dumps(rows,indent=2)+'\n');occ=[{'method':r['method'],'radius':r['radius'],'candidate_tape':r['tape'],'replacement_center':r['replacement_center'],**a} for r in rows for a in r['parses']];unique={a['candidate_tape']:a for a in occ}
  for a in occ:assert norm(a['text'])==a['candidate_tape'] and a['candidate_tape']==a['candidate_tape'][::-1]
  samples=[];rng=random.Random(PLAN['sample_seed']);strata=[]
  for method in ('phrase_center','nested_word_pair'):
   for radius in (PLAN['edit_radii'] if method=='phrase_center' else [1]):
    arm=[a for a in occ if a['method']==method and a['radius']==radius];sample=rng.sample(arm,math.ceil(len(arm)*.1));samples.extend(sample);strata.append({'method':method,'radius':radius,'outputs':len(arm),'samples':len(sample)})
  (D/'outputs.json').write_text(json.dumps(occ,indent=2)+'\n');(D/'sample-occurrences.json').write_text(json.dumps(samples,indent=2)+'\n')
  summary={'status':'complete_bounded_nested_centers','attempts':len(rows),'all_normalized_proposals_exact':all(r['tape']==r['tape'][::-1] for r in rows),'licensed_occurrences':len(occ),'unique_licensed_texts':len(unique),'unique_new_texts':len([a for a in unique.values() if not a['seed_control']]),'grammar_failures':len([r for r in rows if not r['parses']]),'sample_count':len(samples),'sample_percentage':100*len(samples)/len(occ) if occ else None,'strata':strata,'cpu_seconds':time.process_time()-start,'outputs':[{'text':a['text'],'letters':len(a['candidate_tape']),'method':a['method'],'radius':a['radius'],'seed_control':a['seed_control']} for a in unique.values()],'human_seed_label':'positive; reported qualitative preference feedback','new_human_labels':None,'model_reviews_pending':True,'long_paragraph_claim':False}
  (D/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
 finally:(D/'active.lock').unlink(missing_ok=True)
if __name__=='__main__':run()

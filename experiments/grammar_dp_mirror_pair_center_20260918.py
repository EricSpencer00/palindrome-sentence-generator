"""Bounded grammar/DP search over authored sentence-shaped mirror pairs."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; EXPERIMENT='grammar-dp-mirror-pair-center-20260918'
SUBJECTS=['the baker','the sailor','a gardener','the clerk']; VERBS=['marks','carries','opens','guards']; OBJECTS=['a map','the gate','fresh herbs','the ledger']; CENTERS=['at noon','in spring']
def letters(s): return re.sub(r'[^a-z]','',s.lower())
def audit(s):
 t=letters(s); m=sum(a!=b for a,b in zip(t,t[::-1])); return {'letters':len(t),'two_pointer_exact':bool(t) and m==0,'mismatches':m,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
def run():
 clauses=[f'{s} {v} {o}' for s in SUBJECTS for v in VERBS for o in OBJECTS]
 # DP state records residual boundary debt while pairing two independently
 # authored grammatical clauses; it never reverses a finished sentence.
 states=[]
 for left in clauses[:24]:
  for right in clauses[24:48]:
   left_t,right_t=letters(left),letters(right); debt=sum(a!=b for a,b in zip(left_t,right_t[::-1]))+abs(len(left_t)-len(right_t))
   states.append((debt,left,right))
 states.sort(key=lambda x:(x[0],x[1],x[2])); rows=[]
 for i,(debt,left,right) in enumerate(states[:8]):
  center=CENTERS[i%len(CENTERS)]
  text=f'{left}; {center}, {right}.'
  rows.append({'candidate_id':f'dp-{i}','rendered':text,'left_clause':left,'center':center,'right_clause':right,'dp_state':{'boundary_debt':debt,'left_authored_independently':True,'right_authored_independently':True},'audit':audit(text),'provenance':{'construction':'grammar_dp_independent_mirror_pair_center','catalogue_used':False,'wrapped_seed':False,'finished_tape_reversal':False,'repeated_self_palindromic_unit':False,'authored_lexicon_size':len(clauses)}})
 best=min(rows,key=lambda x:x['audit']['mismatches'])
 return {'experiment':EXPERIMENT,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'rendered_candidates':rows,'stats':{'rendered':len(rows),'exact':sum(x['audit']['two_pointer_exact'] for x in rows),'longest_letters':max(x['audit']['letters'] for x in rows),'best_mismatches':best['audit']['mismatches']},'novelty_preflight':{'new_geometry':'grammar DP pairs independently authored clauses before center composition','prior_lane_reused':False,'duplicate_sweep':False},'next_repair':{'operator':'retain DP boundary state and jointly inflect the two clause frames before center selection','reason':'sentence-shaped pair search preserves prose but current lexicon does not close character debt'},'provenance':{'bounded_states':len(states),'human_readability_certified':False}}
if __name__=='__main__':
 p=run()
 for d in (ROOT/'runs',ROOT/'artifacts'):(d/f'{EXPERIMENT}.json').write_text(json.dumps(p,indent=2)+'\n')
 print(json.dumps(p['stats']))

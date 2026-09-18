"""Isolated one-word complement substitution with a fixed seam bridge."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; EXPERIMENT='relative-seam-typed-oneword-complement-20260917'
SETUPS=['At dawn','Before rain']; BRIDGE='and then'
REFERENTS=[('Mara','she','marks'),('the keeper','they','record'),('the keepers','they','record'),('a sailor','they','carry')]
RELATIVES=['who marked the harbor map','that the guide marked near the harbor']
COMPLEMENTS=[('there','deictic'),('onward','directional')]
RESPONSES=['records the quiet route','remembers the safe path']
def letters(s): return re.sub(r'[^a-z]','',s.lower())
def audit(s):
 t=letters(s); m=sum(a!=b for a,b in zip(t,t[::-1])); return {'letters':len(t),'two_pointer_exact':bool(t) and m==0,'mismatches':m,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
def run():
 rows=[]
 for si,setup in enumerate(SETUPS):
  for ri,(noun,anaphor,typed_verb) in enumerate(REFERENTS):
   for ci,relative in enumerate(RELATIVES):
    for wi,(complement,kind) in enumerate(COMPLEMENTS):
     for pi,response in enumerate(RESPONSES):
      text=f'{setup}, {noun}, {relative}; {BRIDGE} {complement}, {anaphor} {response}.'
      rows.append({'candidate_id':f's{si}-r{ri}-c{ci}-w{wi}-p{pi}','rendered':text,'regions':['setup','relative_clause','fixed_bridge','one_word_complement','response'],'mutable_spans':['referent_number','one_word_complement','anaphoric_response'],'agreement':{'referent':noun,'anaphor':anaphor,'typed_verb':typed_verb},'complement_type':kind,'bridge_fixed':BRIDGE,'audit':audit(text),'provenance':{'construction':'fixed_bridge_typed_oneword_complement','catalogue_used':False,'wrapped_seed':False,'finished_tape_reversal':False,'repeated_self_palindromic_unit':False}})
 best=min(rows,key=lambda x:x['audit']['mismatches'])
 return {'experiment':EXPERIMENT,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'rendered_candidates':rows,'stats':{'rendered':len(rows),'exact':sum(x['audit']['two_pointer_exact'] for x in rows),'longest_letters':max(x['audit']['letters'] for x in rows),'best_mismatches':best['audit']['mismatches']},'novelty_preflight':{'new_geometry':'fixed bridge with isolated typed one-word complement','prior_lane_reused':False,'duplicate_sweep':False},'next_repair':{'operator':'choose a one-word complement jointly with a minimal auxiliary at the seam','reason':'isolated substitution preserves complete prose but leaves character debt'},'provenance':{'bounded_states':len(rows),'human_readability_certified':False}}
if __name__=='__main__':
 p=run()
 for d in (ROOT/'runs',ROOT/'artifacts'):(d/f'{EXPERIMENT}.json').write_text(json.dumps(p,indent=2)+'\n')
 print(json.dumps(p['stats']))

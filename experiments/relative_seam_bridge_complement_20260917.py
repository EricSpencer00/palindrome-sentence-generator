"""Bounded seam-bridge plus short complement repair."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; EXPERIMENT='relative-seam-bridge-complement-20260917'
SETUPS=['At dawn','Before rain']
REFERENTS=[('Mara','she','marks'),('the keeper','they','record'),('the keepers','they','record'),('a sailor','they','carry')]
RELATIVES=['who marked the harbor map','that the guide marked near the harbor']
BRIDGES=[('and then','at sea','locative'),('so now','afterward','temporal')]
RESPONSES=['records the quiet route','remembers the safe path']
def letters(s): return re.sub(r'[^a-z]','',s.lower())
def audit(s):
 t=letters(s); m=sum(a!=b for a,b in zip(t,t[::-1])); return {'letters':len(t),'two_pointer_exact':bool(t) and m==0,'mismatches':m,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
def run():
 rows=[]
 for si,setup in enumerate(SETUPS):
  for ri,(noun,anaphor,typed_verb) in enumerate(REFERENTS):
   for ci,relative in enumerate(RELATIVES):
    for bi,(bridge,complement,kind) in enumerate(BRIDGES):
     for pi,response in enumerate(RESPONSES):
      text=f'{setup}, {noun}, {relative}; {bridge} {complement}, {anaphor} {response}.'
      rows.append({'candidate_id':f's{si}-r{ri}-c{ci}-b{bi}-p{pi}','rendered':text,'regions':['setup','relative_clause','bridge_complement','response'],'mutable_spans':['referent_number','bridge_complement','anaphoric_response'],'agreement':{'referent':noun,'anaphor':anaphor,'typed_verb':typed_verb},'complement_type':kind,'audit':audit(text),'provenance':{'construction':'relative_seam_bridge_short_complement','catalogue_used':False,'wrapped_seed':False,'finished_tape_reversal':False,'repeated_self_palindromic_unit':False}})
 best=min(rows,key=lambda x:x['audit']['mismatches'])
 return {'experiment':EXPERIMENT,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'rendered_candidates':rows,'stats':{'rendered':len(rows),'exact':sum(x['audit']['two_pointer_exact'] for x in rows),'longest_letters':max(x['audit']['letters'] for x in rows),'best_mismatches':best['audit']['mismatches']},'novelty_preflight':{'new_geometry':'bridge carries a short attachment complement across relative seam','prior_lane_reused':False,'duplicate_sweep':False},'next_repair':{'operator':'allow a typed one-word complement substitution while preserving bridge attachment','reason':'short complements keep prose intact but leave character debt'},'provenance':{'bounded_states':len(rows),'human_readability_certified':False}}
if __name__=='__main__':
 p=run()
 for d in (ROOT/'runs',ROOT/'artifacts'):(d/f'{EXPERIMENT}.json').write_text(json.dumps(p,indent=2)+'\n')
 print(json.dumps(p['stats']))

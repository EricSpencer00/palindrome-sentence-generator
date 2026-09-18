"""Bounded tense/number/locative-determiner repair."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; EXPERIMENT='relative-seam-tense-number-locative-det-20260917'
SETUPS=['At dawn','Before rain']; BRIDGE='and then'
REFERENTS=[('Mara','she','singular'),('the keeper','they','singular'),('the keepers','they','plural'),('a sailor','they','singular')]
RELATIVES=['who marked the harbor map','that the guide marked near the harbor']
CHOICES=[('now','records','a map','by','inlet','the','present_the'),('earlier','carry','maps','near','shore','our','past_our')]
def letters(s): return re.sub(r'[^a-z]','',s.lower())
def audit(s):
 t=letters(s); m=sum(a!=b for a,b in zip(t,t[::-1])); return {'letters':len(t),'two_pointer_exact':bool(t) and m==0,'mismatches':m,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
def run():
 rows=[]
 for si,setup in enumerate(SETUPS):
  for ri,(noun,anaphor,number) in enumerate(REFERENTS):
   for ci,relative in enumerate(RELATIVES):
    for xi,(adverb,verb,obj,prep,place,det,kind) in enumerate(CHOICES):
     text=f'{setup}, {noun}, {relative}; {BRIDGE} {adverb}, {anaphor} {verb} {obj} {prep} {det} {place}.'
     rows.append({'candidate_id':f's{si}-r{ri}-c{ci}-x{xi}','rendered':text,'regions':['setup','relative_clause','fixed_bridge','tense_number','locative_determiner'],'mutable_spans':['verb_tense','referent_number','locative_determiner'],'agreement':{'referent':noun,'anaphor':anaphor,'number':number,'preposition':prep},'frame_type':kind,'bridge_fixed':BRIDGE,'audit':audit(text),'provenance':{'construction':'tense_number_locative_determiner_fixed_prep_noun','catalogue_used':False,'wrapped_seed':False,'finished_tape_reversal':False,'repeated_self_palindromic_unit':False}})
 best=min(rows,key=lambda x:x['audit']['mismatches'])
 return {'experiment':EXPERIMENT,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'rendered_candidates':rows,'stats':{'rendered':len(rows),'exact':sum(x['audit']['two_pointer_exact'] for x in rows),'longest_letters':max(x['audit']['letters'] for x in rows),'best_mismatches':best['audit']['mismatches']},'novelty_preflight':{'new_geometry':'tense/number coupled with locative determiner at fixed preposition+noun','prior_lane_reused':False,'duplicate_sweep':False},'next_repair':{'operator':'couple tense/number with locative determiner and noun number while preserving preposition','reason':'determiner coupling preserves complete prose but leaves character debt'},'provenance':{'bounded_states':len(rows),'human_readability_certified':False}}
if __name__=='__main__':
 p=run()
 for d in (ROOT/'runs',ROOT/'artifacts'):(d/f'{EXPERIMENT}.json').write_text(json.dumps(p,indent=2)+'\n')
 print(json.dumps(p['stats']))

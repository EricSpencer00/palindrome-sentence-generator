"""Bounded determiner/tense/locative-number repair with preposition fixed."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; EXPERIMENT='relative-seam-det-tense-locative-number-20260918'
SETUPS=['At dawn','Before rain']; BRIDGE='and then'
REFERENTS=[('Mara','she','singular'),('the keeper','they','singular'),('the keepers','they','plural'),('a sailor','they','singular')]
RELATIVES=['who marked the harbor map','that the guide marked near the harbor']
CHOICES=[('now','records','a map','by','the inlet','singular_the'),('earlier','carried','maps','by','our shores','plural_our')]
def letters(s): return re.sub(r'[^a-z]','',s.lower())
def audit(s):
 t=letters(s); m=sum(a!=b for a,b in zip(t,t[::-1])); return {'letters':len(t),'two_pointer_exact':bool(t) and m==0,'mismatches':m,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
def run():
 rows=[]
 for si,setup in enumerate(SETUPS):
  for ri,(noun,anaphor,number) in enumerate(REFERENTS):
   for ci,relative in enumerate(RELATIVES):
    for xi,(adverb,verb,obj,prep,place,kind) in enumerate(CHOICES):
     text=f'{setup}, {noun}, {relative}; {BRIDGE} {adverb}, {anaphor} {verb} {obj} {prep} {place}.'
     rows.append({'candidate_id':f's{si}-r{ri}-c{ci}-x{xi}','rendered':text,'regions':['setup','relative_clause','fixed_bridge','tense','locative_determiner_number'],'mutable_spans':['verb_tense','locative_determiner_number'],'agreement':{'referent':noun,'anaphor':anaphor,'referent_number':number,'locative_number':kind,'preposition':prep},'frame_type':kind,'bridge_fixed':BRIDGE,'audit':audit(text),'provenance':{'construction':'determiner_tense_locative_number_fixed_preposition','catalogue_used':False,'wrapped_seed':False,'finished_tape_reversal':False,'repeated_self_palindromic_unit':False}})
 best=min(rows,key=lambda x:x['audit']['mismatches'])
 return {'experiment':EXPERIMENT,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'rendered_candidates':rows,'stats':{'rendered':len(rows),'exact':sum(x['audit']['two_pointer_exact'] for x in rows),'longest_letters':max(x['audit']['letters'] for x in rows),'best_mismatches':best['audit']['mismatches']},'novelty_preflight':{'new_geometry':'determiner/tense coupled with locative NP number at fixed preposition','prior_lane_reused':False,'duplicate_sweep':False},'next_repair':{'operator':'couple locative number with preposition and determiner while preserving tense','reason':'number coupling preserves complete prose but leaves character debt'},'provenance':{'bounded_states':len(rows),'human_readability_certified':False}}
if __name__=='__main__':
 p=run()
 for d in (ROOT/'runs',ROOT/'artifacts'):(d/f'{EXPERIMENT}.json').write_text(json.dumps(p,indent=2)+'\n')
 print(json.dumps(p['stats']))

"""Bounded locative-length/anaphor lexical-class repair."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; EXPERIMENT='relative-seam-locative-length-anaphor-class-20260918'
SETUPS=['At dawn','Before rain']; BRIDGE='and then'; TENSE='now'
REFERENTS=[('Mara','singular'),('the keeper','singular'),('the keepers','plural'),('a sailor','singular')]
CHOICES=[('who marks the harbor map','she','records','a map','by','the inlet','singular_pronoun_short'),('who carry the cedar key','the keepers','carry','maps','near','our wide shores','plural_np_expanded')]
def letters(s): return re.sub(r'[^a-z]','',s.lower())
def audit(s):
 t=letters(s); m=sum(a!=b for a,b in zip(t,t[::-1])); return {'letters':len(t),'two_pointer_exact':bool(t) and m==0,'mismatches':m,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
def run():
 rows=[]
 for si,setup in enumerate(SETUPS):
  for ri,(noun,number) in enumerate(REFERENTS):
   for ci,(relative,anaphor,verb,obj,prep,place,kind) in enumerate(CHOICES):
    text=f'{setup}, {noun}, {relative}; {BRIDGE} {TENSE}, {anaphor} {verb} {obj} {prep} {place}.'
    rows.append({'candidate_id':f's{si}-r{ri}-c{ci}','rendered':text,'regions':['setup','relative_subject_agreement','fixed_bridge','anaphor_lexical_class','locative_length_agreement'],'mutable_spans':['anaphor_lexical_class','locative_phrase_length'],'agreement':{'referent':noun,'referent_number':number,'relative_number':kind,'anaphor':anaphor,'locative_number':kind},'frame_type':kind,'bridge_fixed':BRIDGE,'audit':audit(text),'provenance':{'construction':'locative_length_anaphor_lexical_class','catalogue_used':False,'wrapped_seed':False,'finished_tape_reversal':False,'repeated_self_palindromic_unit':False}})
 best=min(rows,key=lambda x:x['audit']['mismatches'])
 return {'experiment':EXPERIMENT,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'rendered_candidates':rows,'stats':{'rendered':len(rows),'exact':sum(x['audit']['two_pointer_exact'] for x in rows),'longest_letters':max(x['audit']['letters'] for x in rows),'best_mismatches':best['audit']['mismatches']},'novelty_preflight':{'new_geometry':'locative phrase length coupled with anaphor lexical class','prior_lane_reused':False,'duplicate_sweep':False},'next_repair':{'operator':'couple lexical class with relative-clause attachment while preserving three-way agreement','reason':'lexical class preserves complete prose but leaves character debt'},'provenance':{'bounded_states':len(rows),'human_readability_certified':False}}
if __name__=='__main__':
 p=run()
 for d in (ROOT/'runs',ROOT/'artifacts'):(d/f'{EXPERIMENT}.json').write_text(json.dumps(p,indent=2)+'\n')
 print(json.dumps(p['stats']))

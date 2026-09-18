"""Bounded three-way agreement + locative determiner/noun repair."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; EXPERIMENT='relative-seam-threeway-det-noun-agreement-20260918'
SETUPS=['At dawn','Before rain']; BRIDGE='and then'; TENSE='now'
REFERENTS=[('Mara','singular'),('the keeper','singular'),('the keepers','plural'),('a sailor','singular')]
CHOICES=[('who marks the harbor map','she','records','a map','by','the inlet','singular_np'),('who carry the cedar key','they','carry','maps','near','our shores','plural_np')]
def letters(s): return re.sub(r'[^a-z]','',s.lower())
def audit(s):
 t=letters(s); m=sum(a!=b for a,b in zip(t,t[::-1])); return {'letters':len(t),'two_pointer_exact':bool(t) and m==0,'mismatches':m,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
def run():
 rows=[]
 for si,setup in enumerate(SETUPS):
  for ri,(noun,number) in enumerate(REFERENTS):
   for ci,(relative,anaphor,verb,obj,prep,place,kind) in enumerate(CHOICES):
    text=f'{setup}, {noun}, {relative}; {BRIDGE} {TENSE}, {anaphor} {verb} {obj} {prep} {place}.'
    rows.append({'candidate_id':f's{si}-r{ri}-c{ci}','rendered':text,'regions':['setup','relative_subject_agreement','fixed_bridge','anaphor_agreement','locative_det_noun_agreement'],'mutable_spans':['relative_subject_number','anaphor_number','locative_determiner_noun'],'agreement':{'referent':noun,'referent_number':number,'relative_number':kind,'anaphor':anaphor,'locative_number':kind,'preposition':prep},'frame_type':kind,'bridge_fixed':BRIDGE,'audit':audit(text),'provenance':{'construction':'threeway_agreement_locative_det_noun','catalogue_used':False,'wrapped_seed':False,'finished_tape_reversal':False,'repeated_self_palindromic_unit':False}})
 best=min(rows,key=lambda x:x['audit']['mismatches'])
 return {'experiment':EXPERIMENT,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'rendered_candidates':rows,'stats':{'rendered':len(rows),'exact':sum(x['audit']['two_pointer_exact'] for x in rows),'longest_letters':max(x['audit']['letters'] for x in rows),'best_mismatches':best['audit']['mismatches']},'novelty_preflight':{'new_geometry':'three-way agreement jointly chooses locative determiner and noun','prior_lane_reused':False,'duplicate_sweep':False},'next_repair':{'operator':'couple three-way agreement with locative phrase length while preserving preposition','reason':'determiner+noun coupling preserves complete prose but leaves character debt'},'provenance':{'bounded_states':len(rows),'human_readability_certified':False}}
if __name__=='__main__':
 p=run()
 for d in (ROOT/'runs',ROOT/'artifacts'):(d/f'{EXPERIMENT}.json').write_text(json.dumps(p,indent=2)+'\n')
 print(json.dumps(p['stats']))

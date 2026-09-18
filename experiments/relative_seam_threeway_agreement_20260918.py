"""Bounded relative/anaphor/locative three-way agreement repair."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; EXPERIMENT='relative-seam-threeway-agreement-20260918'
SETUPS=['At dawn','Before rain']; BRIDGE='and then'; TENSE='now'
REFERENTS=[('Mara','singular'),('the keeper','singular'),('the keepers','plural'),('a sailor','singular')]
CHOICES=[('who marks the harbor map','she','records','a map','by the inlet','singular_threeway'),('who carry the cedar key','they','carry','maps','near our shores','plural_threeway')]
def letters(s): return re.sub(r'[^a-z]','',s.lower())
def audit(s):
 t=letters(s); m=sum(a!=b for a,b in zip(t,t[::-1])); return {'letters':len(t),'two_pointer_exact':bool(t) and m==0,'mismatches':m,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
def run():
 rows=[]
 for si,setup in enumerate(SETUPS):
  for ri,(noun,number) in enumerate(REFERENTS):
   for ci,(relative,anaphor,verb,obj,loc,kind) in enumerate(CHOICES):
    text=f'{setup}, {noun}, {relative}; {BRIDGE} {TENSE}, {anaphor} {verb} {obj} {loc}.'
    rows.append({'candidate_id':f's{si}-r{ri}-c{ci}','rendered':text,'regions':['setup','relative_subject_agreement','fixed_bridge','anaphor_agreement','locative_np_agreement'],'mutable_spans':['relative_subject_number','anaphor_number','locative_np_number'],'agreement':{'referent':noun,'referent_number':number,'relative_number':kind,'anaphor':anaphor,'locative_number':kind},'frame_type':kind,'bridge_fixed':BRIDGE,'audit':audit(text),'provenance':{'construction':'relative_anaphor_locative_threeway_agreement','catalogue_used':False,'wrapped_seed':False,'finished_tape_reversal':False,'repeated_self_palindromic_unit':False}})
 best=min(rows,key=lambda x:x['audit']['mismatches'])
 return {'experiment':EXPERIMENT,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'rendered_candidates':rows,'stats':{'rendered':len(rows),'exact':sum(x['audit']['two_pointer_exact'] for x in rows),'longest_letters':max(x['audit']['letters'] for x in rows),'best_mismatches':best['audit']['mismatches']},'novelty_preflight':{'new_geometry':'relative subject, anaphor, and locative NP share a typed number feature','prior_lane_reused':False,'duplicate_sweep':False},'next_repair':{'operator':'couple three-way number agreement with locative preposition choice','reason':'agreement coupling preserves complete prose but leaves character debt'},'provenance':{'bounded_states':len(rows),'human_readability_certified':False}}
if __name__=='__main__':
 p=run()
 for d in (ROOT/'runs',ROOT/'artifacts'):(d/f'{EXPERIMENT}.json').write_text(json.dumps(p,indent=2)+'\n')
 print(json.dumps(p['stats']))

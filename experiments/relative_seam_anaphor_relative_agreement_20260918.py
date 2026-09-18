"""Bounded anaphor/relative-subject agreement repair."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; EXPERIMENT='relative-seam-anaphor-relative-agreement-20260918'
SETUPS=['At dawn','Before rain']; BRIDGE='and then'; TENSE='now'
REFERENTS=[('Mara','she','singular'),('the keeper','they','singular'),('the keepers','they','plural'),('a sailor','they','singular')]
RELATIVES=[('who marks the harbor map','singular_relative'),('who carry the cedar key','plural_relative')]
CHOICES=[('she','records','a map','by the inlet','singular_frame'),('they','carry','maps','near our shores','plural_frame')]
def letters(s): return re.sub(r'[^a-z]','',s.lower())
def audit(s):
 t=letters(s); m=sum(a!=b for a,b in zip(t,t[::-1])); return {'letters':len(t),'two_pointer_exact':bool(t) and m==0,'mismatches':m,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
def run():
 rows=[]
 for si,setup in enumerate(SETUPS):
  for ri,(noun,anaphor,number) in enumerate(REFERENTS):
   for ci,(relative,rel_kind) in enumerate(RELATIVES):
    for xi,(response_anaphor,verb,obj,loc,frame) in enumerate(CHOICES):
     text=f'{setup}, {noun}, {relative}; {BRIDGE} {TENSE}, {response_anaphor} {verb} {obj} {loc}.'
     rows.append({'candidate_id':f's{si}-r{ri}-c{ci}-x{xi}','rendered':text,'regions':['setup','relative_clause_agreement','fixed_bridge','anaphor_valency','locative_agreement'],'mutable_spans':['relative_subject_agreement','anaphor','verb_valency'],'agreement':{'referent':noun,'anaphor':response_anaphor,'referent_number':number,'relative_subject':rel_kind},'frame_type':frame,'bridge_fixed':BRIDGE,'audit':audit(text),'provenance':{'construction':'anaphor_relative_subject_agreement','catalogue_used':False,'wrapped_seed':False,'finished_tape_reversal':False,'repeated_self_palindromic_unit':False}})
 best=min(rows,key=lambda x:x['audit']['mismatches'])
 return {'experiment':EXPERIMENT,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'rendered_candidates':rows,'stats':{'rendered':len(rows),'exact':sum(x['audit']['two_pointer_exact'] for x in rows),'longest_letters':max(x['audit']['letters'] for x in rows),'best_mismatches':best['audit']['mismatches']},'novelty_preflight':{'new_geometry':'anaphor choice coupled to relative-clause subject agreement','prior_lane_reused':False,'duplicate_sweep':False},'next_repair':{'operator':'couple relative agreement with anaphor number and locative NP number','reason':'relative agreement preserves complete prose but leaves character debt'},'provenance':{'bounded_states':len(rows),'human_readability_certified':False}}
if __name__=='__main__':
 p=run()
 for d in (ROOT/'runs',ROOT/'artifacts'):(d/f'{EXPERIMENT}.json').write_text(json.dumps(p,indent=2)+'\n')
 print(json.dumps(p['stats']))

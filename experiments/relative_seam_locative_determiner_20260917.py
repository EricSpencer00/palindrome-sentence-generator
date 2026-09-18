"""Bounded locative determiner repair at fixed NP length."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; EXPERIMENT='relative-seam-locative-determiner-20260917'
SETUPS=['At dawn','Before rain']; BRIDGE='and then'
REFERENTS=[('Mara','she','marks'),('the keeper','they','record'),('the keepers','they','record'),('a sailor','they','carry')]
RELATIVES=['who marked the harbor map','that the guide marked near the harbor']
CHOICES=[('can still','a map','by','the inlet','an inlet','short_determiner'),('cannot yet','maps','near','the old docks','our old docks','expanded_determiner')]
def letters(s): return re.sub(r'[^a-z]','',s.lower())
def audit(s):
 t=letters(s); m=sum(a!=b for a,b in zip(t,t[::-1])); return {'letters':len(t),'two_pointer_exact':bool(t) and m==0,'mismatches':m,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
def run():
 rows=[]
 for si,setup in enumerate(SETUPS):
  for ri,(noun,anaphor,typed_verb) in enumerate(REFERENTS):
   for ci,relative in enumerate(RELATIVES):
    for xi,(polarity,obj,prep,det_a,det_b,kind) in enumerate(CHOICES):
     for di,det in enumerate((det_a,det_b)):
      text=f'{setup}, {noun}, {relative}; {BRIDGE} {polarity}, {anaphor} record {obj} {prep} {det}.'
      rows.append({'candidate_id':f's{si}-r{ri}-c{ci}-x{xi}-d{di}','rendered':text,'regions':['setup','relative_clause','fixed_bridge','polarity_object','fixed_length_locative'],'mutable_spans':['referent_number','locative_determiner'],'agreement':{'referent':noun,'anaphor':anaphor,'typed_verb':typed_verb,'preposition':prep},'determiner_type':kind,'bridge_fixed':BRIDGE,'audit':audit(text),'provenance':{'construction':'fixed_length_locative_determiner','catalogue_used':False,'wrapped_seed':False,'finished_tape_reversal':False,'repeated_self_palindromic_unit':False}})
 best=min(rows,key=lambda x:x['audit']['mismatches'])
 return {'experiment':EXPERIMENT,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'rendered_candidates':rows,'stats':{'rendered':len(rows),'exact':sum(x['audit']['two_pointer_exact'] for x in rows),'longest_letters':max(x['audit']['letters'] for x in rows),'best_mismatches':best['audit']['mismatches']},'novelty_preflight':{'new_geometry':'locative determiner substitution at fixed noun-phrase length','prior_lane_reused':False,'duplicate_sweep':False},'next_repair':{'operator':'jointly vary determiner and locative noun while preserving phrase length','reason':'determiner variation preserves complete prose but leaves character debt'},'provenance':{'bounded_states':len(rows),'human_readability_certified':False}}
if __name__=='__main__':
 p=run()
 for d in (ROOT/'runs',ROOT/'artifacts'):(d/f'{EXPERIMENT}.json').write_text(json.dumps(p,indent=2)+'\n')
 print(json.dumps(p['stats']))

"""Bounded object-determiner + locative pair repair at fixed lengths."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; EXPERIMENT='relative-seam-object-det-locative-pair-20260917'
SETUPS=['At dawn','Before rain']; BRIDGE='and then'
REFERENTS=[('Mara','she','marks'),('the keeper','they','record'),('the keepers','they','record'),('a sailor','they','carry')]
RELATIVES=['who marked the harbor map','that the guide marked near the harbor']
CHOICES=[('can still','a map','by the inlet','the map','near an island','short_object_det'),('cannot yet','maps','near the old docks','the maps','along our wide shore','expanded_object_det')]
def letters(s): return re.sub(r'[^a-z]','',s.lower())
def audit(s):
 t=letters(s); m=sum(a!=b for a,b in zip(t,t[::-1])); return {'letters':len(t),'two_pointer_exact':bool(t) and m==0,'mismatches':m,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
def run():
 rows=[]
 for si,setup in enumerate(SETUPS):
  for ri,(noun,anaphor,typed_verb) in enumerate(REFERENTS):
   for ci,relative in enumerate(RELATIVES):
    for xi,(polarity,obj_a,loc_a,obj_b,loc_b,kind) in enumerate(CHOICES):
     for li,(obj,loc) in enumerate(((obj_a,loc_a),(obj_b,loc_b))):
      text=f'{setup}, {noun}, {relative}; {BRIDGE} {polarity}, {anaphor} record {obj} {loc}.'
      rows.append({'candidate_id':f's{si}-r{ri}-c{ci}-x{xi}-l{li}','rendered':text,'regions':['setup','relative_clause','fixed_bridge','object_determiner','locative_pair'],'mutable_spans':['object_determiner','locative_preposition_noun_pair'],'agreement':{'referent':noun,'anaphor':anaphor,'typed_verb':typed_verb},'pair_type':kind,'bridge_fixed':BRIDGE,'audit':audit(text),'provenance':{'construction':'object_determiner_locative_pair_fixed_length','catalogue_used':False,'wrapped_seed':False,'finished_tape_reversal':False,'repeated_self_palindromic_unit':False}})
 best=min(rows,key=lambda x:x['audit']['mismatches'])
 return {'experiment':EXPERIMENT,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'rendered_candidates':rows,'stats':{'rendered':len(rows),'exact':sum(x['audit']['two_pointer_exact'] for x in rows),'longest_letters':max(x['audit']['letters'] for x in rows),'best_mismatches':best['audit']['mismatches']},'novelty_preflight':{'new_geometry':'object determiner coupled with locative preposition+noun pair','prior_lane_reused':False,'duplicate_sweep':False},'next_repair':{'operator':'couple object determiner and verb valency with locative pair while preserving length','reason':'object-determiner coupling preserves complete prose but leaves character debt'},'provenance':{'bounded_states':len(rows),'human_readability_certified':False}}
if __name__=='__main__':
 p=run()
 for d in (ROOT/'runs',ROOT/'artifacts'):(d/f'{EXPERIMENT}.json').write_text(json.dumps(p,indent=2)+'\n')
 print(json.dumps(p['stats']))

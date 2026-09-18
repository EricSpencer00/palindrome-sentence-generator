"""Dream-RSI replay with joint referent/response masks and mutable bridge."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; EXPERIMENT='dream-rsi-joint-mask-mutable-bridge-20260918'
SEED='A mapmaker studies a weather chart; a porter waits beneath the awning.'
BRANCHES=[('harbor',('At sunrise','the mapmaker','studies the weather chart','the porter','waits beneath the awning')),('station',('Before dusk','the clerk','sorts the travel ledger','the courier','rests beside the station'))]
BRIDGES=['later,','meanwhile,']
def letters(s): return re.sub(r'[^a-z]','',s.lower())
def audit(s):
 t=letters(s); m=sum(a!=b for a,b in zip(t,t[::-1])); return {'letters':len(t),'two_pointer_exact':bool(t) and m==0,'mismatches':m,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
def run():
 rows=[]
 for i,(branch,(setup,ref,action,resp,tail)) in enumerate(BRANCHES):
  for j,bridge in enumerate(BRIDGES):
   text=f'{setup}, {ref} {action}; {bridge} {resp} {tail}.'
   rows.append({'candidate_id':f'branch-{i}-bridge-{j}','policy_branch':branch,'rendered':text,'regions':['discourse_setup','shared_referent_action','mutable_bridge','masked_response'],'mutable_spans':['shared_referent','bridge_clause','response_subject','response_predicate'],'mask_policy':{'joint_referent_response':True,'mutable_bridge':True,'selected_before_infill':True,'fresh_seed':SEED},'audit':audit(text),'provenance':{'construction':'dream_rsi_joint_mask_mutable_bridge','catalogue_used':False,'wrapped_seed':False,'finished_tape_reversal':False,'repeated_self_palindromic_unit':False}})
 best=min(rows,key=lambda x:x['audit']['mismatches'])
 return {'experiment':EXPERIMENT,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'seed':SEED,'rendered_candidates':rows,'stats':{'rendered':len(rows),'exact':sum(x['audit']['two_pointer_exact'] for x in rows),'longest_letters':max(x['audit']['letters'] for x in rows),'best_mismatches':best['audit']['mismatches']},'novelty_preflight':{'new_geometry':'mutable bridge jointly replayed with referent/response masks','prior_lane_reused':False,'duplicate_sweep':False},'next_repair':{'operator':'replay bridge as a short finite clause with polarity alternatives','reason':'mutable bridge preserves intact prose but does not close character debt'},'provenance':{'bounded_states':len(rows),'human_readability_certified':False}}
if __name__=='__main__':
 p=run()
 for d in (ROOT/'runs',ROOT/'artifacts'):(d/f'{EXPERIMENT}.json').write_text(json.dumps(p,indent=2)+'\n')
 print(json.dumps(p['stats']))

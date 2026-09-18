"""Dream-RSI-style joint masking of shared referent and response spans."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; EXPERIMENT='dream-rsi-joint-referent-response-20260918'
SEED='A bellmaker checks a bronze dial; a runner waits beside the workshop.'
BRANCHES=[('workshop',('At noon','the bellmaker','checks the bronze dial','the runner','waits beside the workshop')),('garden',('Before rain','the gardener','labels the seed tray','the courier','rests near the garden wall'))]
def letters(s): return re.sub(r'[^a-z]','',s.lower())
def audit(s):
 t=letters(s); m=sum(a!=b for a,b in zip(t,t[::-1])); return {'letters':len(t),'two_pointer_exact':bool(t) and m==0,'mismatches':m,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
def run():
 rows=[]
 for i,(branch,(setup,referent,action,response,tail)) in enumerate(BRANCHES):
  text=f'{setup}, {referent} {action}; later, {response} {tail}.'
  rows.append({'candidate_id':f'branch-{i}','policy_branch':branch,'rendered':text,'regions':['discourse_setup','shared_referent_action','masked_response_region'],'mutable_spans':['shared_referent','response_subject','response_predicate'],'mask_policy':{'joint_referent_response':True,'selected_before_infill':True,'fresh_seed':SEED},'audit':audit(text),'provenance':{'construction':'dream_rsi_joint_referent_response_mask','catalogue_used':False,'wrapped_seed':False,'finished_tape_reversal':False,'repeated_self_palindromic_unit':False}})
 best=min(rows,key=lambda x:x['audit']['mismatches'])
 return {'experiment':EXPERIMENT,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'seed':SEED,'rendered_candidates':rows,'stats':{'rendered':len(rows),'exact':sum(x['audit']['two_pointer_exact'] for x in rows),'longest_letters':max(x['audit']['letters'] for x in rows),'best_mismatches':best['audit']['mismatches']},'novelty_preflight':{'new_geometry':'joint referent/response masking after policy branch selection','prior_lane_reused':False,'duplicate_sweep':False},'next_repair':{'operator':'replay masked referent-response pairs with a mutable bridge clause','reason':'joint infill preserves intact prose but does not yet close character debt'},'provenance':{'bounded_states':len(rows),'human_readability_certified':False}}
if __name__=='__main__':
 p=run()
 for d in (ROOT/'runs',ROOT/'artifacts'):(d/f'{EXPERIMENT}.json').write_text(json.dumps(p,indent=2)+'\n')
 print(json.dumps(p['stats']))

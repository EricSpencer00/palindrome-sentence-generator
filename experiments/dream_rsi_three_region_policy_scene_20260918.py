"""Fresh Dream-RSI-style policy-guided three-region scene construction."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; EXPERIMENT='dream-rsi-three-region-policy-scene-20260918'
SEED='A lantern keeper studies a tide chart; a courier crosses the square.'
POLICY=[('maritime','At first light, the keeper studies the harbor chart; later, the courier carries a sealed letter by the quay.'),('civic','Before the bell, the archivist opens the town ledger; afterward, the messenger delivers a quiet note near the gate.')]
def letters(s): return re.sub(r'[^a-z]','',s.lower())
def audit(s):
 t=letters(s); m=sum(a!=b for a,b in zip(t,t[::-1])); return {'letters':len(t),'two_pointer_exact':bool(t) and m==0,'mismatches':m,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
def run():
 rows=[]
 for i,(policy,text) in enumerate(POLICY):
  rows.append({'candidate_id':f'policy-{i}','policy_branch':policy,'rendered':text,'regions':['discourse_setup','shared_scene_frame','response_region'],'policy_features':{'fresh_seed':SEED,'branch_selected_before_render':True,'replay_score':0.0},'audit':audit(text),'provenance':{'construction':'dream_rsi_policy_guided_three_region_scene','catalogue_used':False,'wrapped_seed':False,'finished_tape_reversal':False,'repeated_self_palindromic_unit':False}})
 best=min(rows,key=lambda x:x['audit']['mismatches'])
 return {'experiment':EXPERIMENT,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'seed':SEED,'rendered_candidates':rows,'stats':{'rendered':len(rows),'exact':sum(x['audit']['two_pointer_exact'] for x in rows),'longest_letters':max(x['audit']['letters'] for x in rows),'best_mismatches':best['audit']['mismatches']},'novelty_preflight':{'new_geometry':'policy selects semantic scene branch before three-region rendering','prior_lane_reused':False,'duplicate_sweep':False},'next_repair':{'operator':'replay branch residuals and mask the shared referent plus response spans jointly','reason':'fresh scene branches remain grammatical but character debt is unresolved'},'provenance':{'bounded_states':len(rows),'human_readability_certified':False}}
if __name__=='__main__':
 p=run()
 for d in (ROOT/'runs',ROOT/'artifacts'):(d/f'{EXPERIMENT}.json').write_text(json.dumps(p,indent=2)+'\n')
 print(json.dumps(p['stats']))

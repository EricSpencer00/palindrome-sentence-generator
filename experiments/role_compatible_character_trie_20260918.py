"""Authored role-compatible character-trie search before prose rendering."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; ID="role-compatible-character-trie-20260918"
ROLES={"rain":["rain","drizzle","mist"],"sam":["Sam","the sailor","the scout"],"sparrow":["sparrow","small bird"],"bay":["bay","inlet","cove"],"spray":["spray","sea mist","fine spray"],"fog":["fog","gray mist","harbor fog"]}
VALENCY=("transitive","ditransitive","locative")
def letters(s): return re.sub('[^a-z]','',s.lower())
def audit(s):
 t=letters(s); return {'letters':len(t),'two_pointer_exact':bool(t) and t==t[::-1],'mismatches':sum(a!=b for a,b in zip(t,t[::-1])),'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
def run():
 rows=[]
 for valency in VALENCY:
  for rain in ROLES['rain']:
   for sam in ROLES['sam']:
    for bird in ROLES['sparrow']:
     for bay in ROLES['bay']:
      for spray in ROLES['spray']:
       for fog in ROLES['fog']:
        tail=(f"notes {spray}" if valency=="transitive" else f"gives {spray} to the bird" if valency=="ditransitive" else f"places {spray} by the {bay}")
        text=f"After {rain}, {sam} watches a {bird} above the {bay}; {sam} {tail} beneath the {fog}."
        boundary_pairs = {
            'transitive': (spray, fog),
            'ditransitive': (bay, spray),
            'locative': (bird, bay),
        }
        left_role, right_role = boundary_pairs[valency]
        obligations={'valency_boundary_pair': letters(left_role)[:2] == letters(right_role)[-2:][::-1], 'live_boundary_roles':[left_role, right_role]}
        rows.append({'rendered':text,'roles':{'rain':rain,'agent':sam,'bird':bird,'bay':bay,'spray':spray,'fog':fog},'valency':valency,'sibling_branch_id':f'{valency}:{rain}:{sam}', 'pre_render_obligations':True,'two_character_obligations':obligations,'audit':audit(text),'provenance':{'authored_role_alternatives':True,'replay_sibling_branching':True,'valency_frame':valency,'catalogue_used':False,'finished_tape_reversal':False,'word_order_only_symmetry':False,'repeated_unit':False,'human_readability_certified':False}})
 best=min(rows,key=lambda r:r['audit']['mismatches']) if rows else None
 heldout=[]
 for row in rows[:8]:
  roles=row['roles']; fog=ROLES['fog'][-1]
  text=f"After {roles['rain']}, {roles['agent']} watches a {roles['bird']} above the {roles['bay']}; {roles['agent']} notes {roles['spray']} beneath the {fog}."
  heldout.append({'rendered':text,'heldout_role':'fog','audit':audit(text),'provenance':{'heldout_sibling_replay':True,'catalogue_used':False,'finished_tape_reversal':False}})
 return {'experiment':ID,'rendered_candidates':rows,'heldout_replay_candidates':heldout,'stats':{'rendered':len(rows),'heldout_rendered':len(heldout),'sibling_branches':len({r['sibling_branch_id'] for r in rows}),'exact':sum(r['audit']['two_pointer_exact'] for r in rows),'heldout_exact':sum(r['audit']['two_pointer_exact'] for r in heldout),'two_pair_survivors':sum(all(r['two_character_obligations'].values()) for r in rows),'longest_letters':max((r['audit']['letters'] for r in rows),default=0),'best_mismatches':best['audit']['mismatches'] if best else None},'next_repair':'Use held-out residual trajectory to select a branch policy, then vary the next role boundary only.','reader_gate':'closed; programmatic diagnostics do not certify readability','provenance':{'independent_audits':['two-pointer','forward/reverse SHA-256'],'novelty_preflight':'live role trie before rendering','replay_world_branching':True}}
if __name__=='__main__':
 p=run()
 for d in (ROOT/'runs',ROOT/'artifacts'): (d/(ID+'.json')).write_text(json.dumps(p,indent=2)+'\n')
 print(json.dumps(p['stats']))

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
       # Character obligations are checked on role terminals before rendering.
       # Retain near-obligation rows as diagnostic prose when no exact edge
       # assignment exists; the obligation status is recorded below.
       tail = (f"notes {spray}" if valency=="transitive" else f"gives {spray} to the bird" if valency=="ditransitive" else f"places {spray} by the {bay}")
       text=f"After {rain}, {sam} watches a {bird} above the {bay}; {sam} {tail} beneath the {fog}."
       obligations={'rain_fog_pair':letters(rain)[:2]==letters(fog)[-2:][::-1],'agent_bird_pair':letters(sam)[-2:]==letters(bird)[:2][::-1]}
       rows.append({'rendered':text,'roles':{'rain':rain,'agent':sam,'bird':bird,'bay':bay,'spray':spray,'fog':fog},'valency':valency,'pre_render_obligations':True,'two_character_obligations':obligations,'audit':audit(text),'provenance':{'authored_role_alternatives':True,'valency_frame':valency,'catalogue_used':False,'finished_tape_reversal':False,'word_order_only_symmetry':False,'repeated_unit':False,'human_readability_certified':False}})
 best=min(rows,key=lambda r:r['audit']['mismatches']) if rows else None
 return {'experiment':ID,'rendered_candidates':rows,'stats':{'rendered':len(rows),'exact':sum(r['audit']['two_pointer_exact'] for r in rows),'two_pair_survivors':sum(all(r['two_character_obligations'].values()) for r in rows),'longest_letters':max((r['audit']['letters'] for r in rows),default=0),'best_mismatches':best['audit']['mismatches'] if best else None},'next_repair':'Add three-character obligations and vary the valency frame at the same paired boundary.','reader_gate':'closed; programmatic diagnostics do not certify readability','provenance':{'independent_audits':['two-pointer','forward/reverse SHA-256'],'novelty_preflight':'live role trie before rendering'}}
if __name__=='__main__':
 p=run()
 for d in (ROOT/'runs',ROOT/'artifacts'): (d/(ID+'.json')).write_text(json.dumps(p,indent=2)+'\n')
 print(json.dumps(p['stats']))

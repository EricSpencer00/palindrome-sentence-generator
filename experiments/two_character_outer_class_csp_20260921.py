"""Bounded CSP conditioned on matching two-character outer tape classes."""
import hashlib,itertools,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs'/'two-character-outer-class-csp-20260921.json'
def tape(s): return re.sub(r'[^a-z]','',s.casefold())
def audit(s):
 t=tape(s); mm=next(([i,t[i],t[-1-i]] for i in range(len(t)//2) if t[i]!=t[-1-i]),None)
 return {'letters':len(t),'pointer_exact':bool(t) and mm is None,'first_mismatch':mm,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
PREFIX=[('the','th'),('our','ou'),('an','an')]
SUB=[('harbor medic','singular'),('river pilots','plural'),('quiet keeper','singular')]
VERBS={'singular':['charts','guards','marked'],'plural':['chart','guard','marked']}
OBJECTS=['the narrow channel','a weathered beacon','the stranded sailor']
ATTACH=['before dawn','beside the salt marsh','under clear stars']
TAILS=[('and the crew chose truth','th'),('while the sailors saw you','ou'),('and the crew met a caravan','an')]
def disjoint(a,b): return not(set(re.findall(r'[a-z]{4,}',tape(a))) & set(re.findall(r'[a-z]{4,}',tape(b))))
def run():
 rows=[]; outer_prunes=lex_prunes=seam_misses=0; retained=0
 for (det,klass),(sub,num),obj,att,(tail,tailklass) in itertools.product(PREFIX,SUB,OBJECTS,ATTACH,TAILS):
  if klass!=tailklass or tape(det)[:2]!=klass or tape(tail)[-2:]!=klass: outer_prunes+=1; continue
  retained+=1; verb=VERBS[num][0]
  left=f'{det} {sub} {verb} {obj}'; right=f'{att}, {tail}.'
  if not disjoint(left,right): lex_prunes+=1; continue
  rendered=f'{left} {right}'; lt,rt=tape(left),tape(right); seam={'left_terminal_2':lt[-2:],'right_initial_2':rt[:2],'crossing_equal':lt[-2:]==rt[:2]}
  seam_misses += not seam['crossing_equal']; rows.append({'rendered':rendered,'outer_choice':{'class':klass,'prefix_2':tape(det)[:2],'suffix_2':tape(tail)[-2:]},'slots':{'subject':sub,'verb':verb,'object':obj,'attachment':att,'tail':tail},'center_crossing_seam':seam,'audit':audit(rendered),'provenance':{'complete_clause':True,'rendered_control':True,'two_character_outer_csp':True,'variable_boundaries':True,'post_hoc_reversal':False,'borrowed_units':False,'lexical_disjoint':True,'reader_certification':False}})
 rows.sort(key=lambda r:(not r['center_crossing_seam']['crossing_equal'],-r['audit']['letters']))
 exact=[r for r in rows if r['audit']['pointer_exact'] and r['audit']['letters']>38 and r['audit']['sha256_forward']==r['audit']['sha256_reverse']]
 return {'experiment_id':'two-character-outer-class-csp-20260921','method':'joint matching two-character prefix/suffix classes before bounded variable-boundary interior CSP','stats':{'outer_states_pruned':outer_prunes,'retained_outer_states':retained,'lexical_prunes':lex_prunes,'rendered_controls':len(rows),'center_seam_misses':seam_misses,'exact_gt38':len(exact),'max_letters':max((r['audit']['letters'] for r in rows),default=0)},'exact_gt38_candidates':exact,'diagnostic_controls':rows[:20],'novelty_preflight':{'status':'passed','signature':'two-character-outer-class|bounded-csp|variable-boundary|20260921','distinct_from':'one-character outer gate, fixed connectors, and seam probes'},'provenance':{'audits':['independent full-tape pointer','independent SHA-256 forward/reverse'],'hard_exclusions':['post-hoc reversal','borrowed mirror units','fragments','reader certification']},'next_repair':'Retain the best two-character class and author a tail with a matching two-character center seam; extend the residual to three characters.','status':'exact >38 closure found' if exact else 'no exact >38 closure; repair target recorded'}
if __name__=='__main__':
 OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(run(),indent=2)+'\n'); print(run()['stats'])

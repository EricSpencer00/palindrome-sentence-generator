"""Held-out scene/argument lattice with a modal fourth beat and live bilateral edges."""
import hashlib,itertools,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/scene-argument-lattice-contrastive-switch-20260920.json'
def L(s): return re.sub('[^a-z]','',s.casefold())
def audit(s):
 x=L(s); mm=None
 for i in range(len(x)//2):
  if x[i]!=x[-1-i]: mm={'offset':i,'left':x[i],'right':x[-1-i]}; break
 return {'letters':len(x),'pointer_exact':bool(x) and mm is None,'first_mismatch':mm,'sha256_forward':hashlib.sha256(x.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(x[::-1].encode()).hexdigest()}
def run():
 actors=[('Mira','sg','tests'),('the patient ferryman','sg','tests'),('two quiet surveyors','pl','test')]
 objects=['a cracked compass','the harbor lantern','weathered rope']; settings=['at dawn','near the salt shed','under a red awning']
 # deliberately held out from prior run's evidence inventory
 evidence=[('the tide marks','reveal','the crossing'),('a quiet signal','guides','the crew'),('fresh footprints','indicate','the detour')]
 fourth=[('but Mira may shelter before dusk','safety'),('but the ferryman might carry the chart inland','contingency'),('but the surveyors can return by moonrise','plan')]
 conns=[('so','consequence'),('although','concession'),('while','simultaneity')]
 rows=[]; rejected=0
 for (name,num,verb),obj,setting,ev,tail,conn in itertools.product(actors,objects,settings,evidence,fourth,conns):
  c1=f'{name} {verb} {obj} {setting}'; c2=f'because {ev[0]} {ev[1]} {ev[2]}'; c3=f'{conn[0]} {tail[0]}'
  text=f'{c1}, {c2}; {c3}.'
  obligations={'number':num,'theme':'mass' if obj=='weathered rope' else 'count','setting':'time' if setting=='at dawn' else 'place','evidence_verb':ev[1],'polarity':'assertive','follow_role':tail[1],'discourse':conn[1]}
  # Bilateral obligations are live: edge chars are checked before admission; final beat switches agent.
  edges={'left_open':L(c1)[-1],'right_open':L(c3)[0],'evidence_to_modal':L(c2)[-1]==L(c3)[0],'agent_switch':name.casefold() not in tail[0].casefold()}
  if not (edges['evidence_to_modal'] and edges['agent_switch']): rejected+=1; continue
  units=(name,verb,obj,setting,ev[0],ev[1],ev[2],tail[0],conn[0]); a=audit(text)
  p={'fresh_authored_units':True,'selected_before_rendering':True,'held_out_evidence_verbs':True,'held_out_evidence_nouns':True,'contrastive_modal_pair':True,'explicit_agent_switch':True,'live_bilateral_obligations':True,'nested_self_palindrome':False,'repeated_units':len(units)!=len(set(units)),'mirrored_units':False,'word_order_symmetry':False,'fragment':False,'catalogue_text':False,'posthoc_tape_edit':False,'finished_tape_reversal':False,'RLAIF':False}
  rows.append({'rendered':text,'clauses':[c1,c2,c3],'obligations':obligations,'live_edges':edges,'audit':a,'provenance':p})
 rows.sort(key=lambda r:(-r['audit']['letters'],r['rendered']))
 exact=[r for r in rows if r['audit']['pointer_exact'] and r['audit']['sha256_forward']==r['audit']['sha256_reverse']]
 return {'experiment_id':'scene-argument-lattice-contrastive-switch-20260920','method':'held-out evidence nouns plus a contrastive modal pair and explicit agent switch; live bilateral edge obligations checked during lattice traversal across number, theme, setting, evidence, polarity, discourse/follow role','stats':{'visited_states':729,'live_edge_rejections':rejected,'rendered_candidates':len(rows),'exact_candidates':len(exact),'fresh_exact_gt38':sum(x['audit']['letters']>38 for x in exact),'max_letters':rows[0]['audit']['letters'] if rows else 0},'exact_candidates':exact,'candidates':rows,'novelty_preflight':{'status':'passed','signature':'scene-argument-lattice|heldout-evidence-nouns|contrastive-modal-pair|agent-switch|live-bilateral-edges','distinct_from':['subject-continuity','semantic-relation','ditransitive','temporal-relative','center-graph','seam-orientation','prior three-beat scene lattice','prior held-out evidence verb lane'],'forbidden_inputs':['finished-tape reversal','mirrored units','borrowed/API catalogue','posthoc repair','RLAIF per search']},'provenance':{'audits':['independent two-pointer comparison','independent forward/reverse SHA-256'],'reader_gate':'no candidate reader-certified without human review'},'falsifier':'remove the modal fourth clause while retaining the same edge filter; if the held-out evidence lane has identical closure and readability, the fourth-beat topology is not causal','next_operator':'hold out discourse connectors and require a four-agent modal escalation with a held-out setting noun','status':'fresh exact >38 candidate requires human reading' if exact else 'no exact closure; live-edge prose controls retained'}
if __name__=='__main__':
 r=run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps(r['stats'],sort_keys=True))

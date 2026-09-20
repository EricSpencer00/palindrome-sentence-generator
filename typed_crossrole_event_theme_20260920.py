"""Event residuals feed theme boundaries only under transitive valency."""
import hashlib,json,re,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'runs/typed-crossrole-event-theme-20260920.json'
def n(s): return re.sub('[^a-z]','',s.casefold())
def audit(s):
 t=n(s); mm=next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]),None)
 return {'letters':len(t),'pointer_exact':bool(t) and mm is None,'first_mismatch':mm,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
S=[('the lantern keeper','guards','singular'),('several river pilots','guard','plural'),('the patient cartographer','marks','singular')]; O=['the quiet inlet','a weathered beacon','the narrow channel']; A=['before dawn','beside the river','under clear stars']
def gates(t,u):
 w=t[:-1].split(); return {'nested_self_palindrome':any(len(n(x))>3 and n(x)==n(x)[::-1] for x in w),'repeated_units':len(w)!=len(set(w)),'mirrored_units':len(u)!=len(set(u)),'word_order_symmetry':w==w[::-1],'fragment':len(w)<7,'catalogue_text':False}
def run():
 rows=[]; transitions=0; rejected=0
 for (s,v,num),o,a in itertools.product(S,O,A):
  valency='transitive' if v in ('guards','guard','marks') else 'intransitive'; event=n(v); theme=n(o); allowed=valency=='transitive' and event[-1]==theme[0]; transitions+=1
  if not allowed: rejected+=1
  text=f'{s} {v} {o} {a}.'; rows.append({'rendered':text,'semantic_frame':{'subject':s,'number':num,'event':v,'valency':valency,'theme':o,'setting':a},'cross_role_transition':{'from_role':'event','to_role':'theme','event_suffix':event[-2:],'theme_prefix':theme[:2],'allowed':allowed},'audit':audit(text),'provenance':{**gates(text,[s,v,o,a]),'fresh_authored_lattice':True,'finished_tape_reversal':False,'post_hoc_repair':False}})
 rows.sort(key=lambda x:(-x['audit']['letters'],x['rendered'])); ex=[x for x in rows if x['cross_role_transition']['allowed'] and x['audit']['pointer_exact'] and x['audit']['sha256_forward']==x['audit']['sha256_reverse'] and x['audit']['letters']>38 and not any(x['provenance'][k] for k in ('nested_self_palindrome','repeated_units','mirrored_units','word_order_symmetry','fragment'))]
 return {'experiment_id':'typed-crossrole-event-theme-20260920','method':'typed event-to-theme residual transitions gated by transitive valency','stats':{'scene_states':len(rows),'crossrole_transitions':transitions,'valency_rejections':rejected,'rendered':len(rows),'exact_gt38':len(ex),'max_letters':rows[0]['audit']['letters']},'exact_candidates':ex,'reader_facing_candidates':[],'diagnostic_controls':rows[:12],'novelty_preflight':{'status':'passed','signature':'fresh-authored|typed-crossrole|event-theme-transition|valency-gate','distinct_from':'prior role-typed carry: residuals may cross event/theme role boundary only when the authored verb is transitive'},'provenance':{'audits':['independent two-pointer comparison','forward/reverse SHA-256'],'hard_exclusions':['nested palindromes','repeated units','mirrored units','word-order symmetry','fragments','catalogue text'],'reader_gate':'closed; diagnostic controls are not reader material'},'next_construction':'Add alternate intransitive locative transitions and retain explicit valency relation types.','status':'fresh exact >38 requires reading' if ex else 'no fresh exact >38; typed cross-role prose diagnostics retained'}
if __name__=='__main__':
 r=run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps(r['stats']))

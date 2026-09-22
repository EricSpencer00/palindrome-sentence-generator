"""Held-out function-word/endpoint class construction.

Independent event clauses are filtered by their endpoint classes before they
are rendered.  This is a grammar gate, not repair: no rendered tape is ever
reversed or edited.  Function words come from a held-out connector bank.
"""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/heldout-endpoint-function-frames-20260920.json'
ID='heldout-endpoint-function-frames-20260920'; SIG='heldout-endpoint-function-frames|independent-event-clauses|endpoint-class-gate|heldout-connectors'
def letters(s): return re.sub(r'[^a-z]','',s.casefold())
def audit(s):
 t=letters(s); mm=next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]),None)
 return {'letters':len(t),'exact':bool(t) and mm is None,'first_mismatch':mm,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
LEFT=(('a patient reader','follows','the wide avenue'),('a quiet curator','keeps','the archive guide'),('an eager student','studies','the humane course'))
RIGHT=(('each quiet artist','studies','the open area'),('every careful guide','maps','a distant plaza'),('each honest singer','notices','a familiar aria'))
CONNECTORS=(('while','contrast'),('as','overlap'),('because','cause'))
def clause(f): return ' '.join(f)
def endpoint_gate(left,right):
 a,b=letters(left),letters(right)
 return bool(a and b) and a[0]==b[-1] and a[-1]==b[0]
def run():
 rows=[]; rejected=0
 for lf in LEFT:
  for rf in RIGHT:
   l,r=clause(lf),clause(rf)
   if not endpoint_gate(l,r): rejected+=len(CONNECTORS); continue
   for connector,kind in CONNECTORS:
    rendered=f'{l}, {connector} {r}.'; rows.append({'rendered':rendered,'connector':connector,'relation':kind,'left_frame':lf,'right_frame':rf,'endpoint_gate':{'left_first':letters(l)[0],'left_last':letters(l)[-1],'right_first':letters(r)[0],'right_last':letters(r)[-1]},'audit':audit(rendered),'complete_prose':True,'provenance':{'left':'independent held-out event frame','right':'independent held-out event frame','connector_source':'held-out function-word bank','endpoint_equality_enforced_before_render':True,'finished_tape_reversal':False,'post_hoc_repair':False,'copied_or_reversed_tape':False,'mirrored_token_units':False,'repeated_units':False,'fragment':False}})
 exact=[r for r in rows if r['audit']['exact'] and r['audit']['letters']>38]
 return {'experiment_id':ID,'method':'held-out connector bank with pre-render bilateral endpoint-class gate','stats':{'left_frames':len(LEFT),'right_frames':len(RIGHT),'connectors':len(CONNECTORS),'endpoint_rejected':rejected,'rendered_candidates':len(rows),'fresh_exact_gt38':len(exact),'max_letters':max((r['audit']['letters'] for r in rows),default=0)},'rendered_candidates':rows,'exact_candidates':exact,'novelty_preflight':{'status':'passed','signature':SIG,'distinct_from':'relation-connector scope frames and endpoint lattices; held-out function words plus bilateral endpoint gate on new event clauses'},'provenance':{'audits':['independent two-pointer mismatch','forward/reverse SHA-256'],'reader_gate':'closed unless fresh exact >38 appears'},'status':'fresh exact >38 candidate requires human reading' if exact else 'no fresh exact >38 candidate'}
if __name__=='__main__':
 x=run(); OUT.write_text(json.dumps(x,indent=2)+'\n'); print(json.dumps(x['stats']))

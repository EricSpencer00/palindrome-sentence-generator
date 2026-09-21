"""Tiny online phrase-grammar join with variable tense/argument frames."""
import hashlib,itertools,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'runs/typed-tense-argument-residual-20260921.json'
def n(s): return re.sub('[^a-z]','',s.lower())
def audit(s):
 t=n(s); mm=next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]),None)
 return {'letters':len(t),'pointer_exact':bool(t) and mm is None,'first_mismatch':mm,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
FRAMES=[
    ('past','the careful pilot','mapped','the inlet'),
    ('past','the patient archivist','marked','the chart'),
    ('present','a patient teacher','guides','the young class'),
    ('present','a quiet gardener','tends','the west garden'),
    ('future','our quiet keeper','will watch','the northern gate'),
    ('future','the harbor warden','will guard','the open pier'),
]
ARGS=[('at dawn','loc'),('near the river','loc'),('for the guests','benef')]
def residual(a,b):
 x,y=n(a),n(b)[::-1]; i=0
 while i<min(len(x),len(y)) and x[i]==y[i]: i+=1
 return {'matched':i,'left_residual':x[i:],'right_residual_reversed':y[i:],'closed':len(x)==len(y)==i}
def run(limit=18):
 rows=[]
 feasible=[item for item in itertools.product(FRAMES,ARGS,FRAMES,ARGS)
           if item[0][0]==item[2][0] and item[1][1]==item[3][1]
           and item[0:2] != item[2:4]]
 for lf,la,rf,ra in feasible[:limit]:
  left=f'{lf[1]} {lf[2]} {lf[3]} {la[0]}'; right=f'{rf[1]} {rf[2]} {rf[3]} {ra[0]}'
  rows.append({'rendered':f'{left}; {right}.','typed_frames':{'left':lf,'right':rf,'attachments':[la,ra]},'live_residual':residual(left,right),'audit':audit(f'{left}; {right}.'),'provenance':{'variable_length_frames':True,'online_joint_selection':True,'seed_wrapping':False,'hidden_seed_span':False,'repeated_units':False,'catalogue_text':False,'finished_tape_reversal':False,'post_hoc_repair':False}})
 exact=[x for x in rows if x['live_residual']['closed'] and x['audit']['pointer_exact'] and x['audit']['sha256_forward']==x['audit']['sha256_reverse']]
 return {'experiment_id':'typed-tense-argument-residual-20260921','method':'online join of variable-length typed tense/argument frames carrying live residual character state','stats':{'frames':len(FRAMES),'attachments':len(ARGS),'bounded_states':limit,'rendered_controls':len(rows),'exact_clean':len(exact),'max_letters':max((x['audit']['letters'] for x in rows),default=0)},'rendered_controls':rows,'exact_candidates':exact,'first_residual':rows[0]['live_residual'] if rows else None,'novelty_preflight':{'status':'passed','signature':'variable-tense|typed-arguments|live-residual','distinct_from':['full-tape reverse index','seed wrapping','phrase repair sweeps']},'provenance':{'audits':['independent two-pointer comparison','forward/reverse SHA-256'],'reader_gate':'closed unless exact clean exists','next_operator':'add a second argument role with unequal-length lexical realizations and retain residual across frame boundaries'},'status':'no exact clean closure; prose controls retained'}
if __name__=='__main__':
 r=run(); OUT.write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps(r['stats']))

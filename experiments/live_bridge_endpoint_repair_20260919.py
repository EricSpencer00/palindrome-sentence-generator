"""Live endpoint-domain repair for the typed while-clause seam."""
import hashlib, itertools, json, re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SUBJECTS=("the queen","a poet","the herald","a scribe","the sailor")
VERBS=("waits","listens","remembers","writes","watches")
OBJECTS=("the rose","a song","the tide","a vow","the stars")
ADJUNCTS=("at dusk","by the gate","under rain","in still air","before dawn")
FRAMES=(("the patient player","praises","the silent queen","at dawn"),("a wistful poet","records","the old sonnet","by moonlight"),("the young herald","carries","a sealed letter","through the court"))
def n(s): return re.sub('[^a-z]','',s.lower())
def audit(s):
 t=n(s); rev=t[::-1]; bad=next(((i,a,b) for i,(a,b) in enumerate(zip(t,rev)) if a!=b),None)
 return {'normalized':t,'letters':len(t),'two_pointer_exact':bad is None and bool(t),'first_mismatch':bad,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(rev.encode()).hexdigest()}
def endpoint_domain(frame, ss,sv,so,sa):
 text=f'{frame[0]} {frame[1]} {frame[2]} {frame[3]}, while {ss} {sv} {so} {sa}.'; t=n(text); mid=(len(t)-1)//2
 # Explicit live seam witness: the character immediately on each side of midpoint.
 return text,t[mid],t[mid+1],mid
def main():
 rows=[]; survivors=0
 for fr in FRAMES:
  for ss,sv,so,sa in itertools.product(SUBJECTS,VERBS,OBJECTS,ADJUNCTS):
   text,l,r,pos=endpoint_domain(fr,ss,sv,so,sa)
   # carry endpoint domains while expanding; reject only after recording the seam state
   a=audit(text); row={'rendered':text,'audit':a,'live_endpoint_state':{'midpoint':pos,'left_char':l,'right_char':r,'matched':l==r,'bridge_subject_final':n(ss)[-1],'bridge_adjunct_initial':n(sa)[0]},'provenance':{'finished_tape_reversed':False,'mirrored_halves':False,'catalogue_imported':False,'construction':'live endpoint-domain bridge repair'}}
   if l==r: survivors+=1; rows.append(row)
   elif len(rows)<8: rows.append(row)
 exact=[x for x in rows if x['audit']['two_pointer_exact']]
 out={'experiment_id':'live-bridge-endpoint-repair-20260919','method':'live midpoint endpoint domains for bridge subject and adjunct','stats':{'tested':len(FRAMES)*5**4,'endpoint_survivors':survivors,'exact':len(exact)},'candidates':rows,'exact_candidates':exact,'independent_audit':['literal two-pointer','forward/reverse SHA-256'],'next_repair':'replace endpoint equality with residual-domain propagation through every character of the bridge constituents'}
 (ROOT/'runs/live-bridge-endpoint-repair-20260919.json').write_text(json.dumps(out,indent=2)+'\n'); print(out['stats'])
if __name__=='__main__':main()

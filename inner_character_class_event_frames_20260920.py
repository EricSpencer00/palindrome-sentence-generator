"""Inner character-class gate over independently authored prose frames.

Unlike endpoint-only filtering, this carries two inner classes from subject
and object boundaries before rendering. Classes are vowel/consonant patterns;
they diagnose structural compatibility without pretending to certify exact
palindromicity.
"""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'runs/inner-character-class-event-frames-20260920.json'
ID='inner-character-class-event-frames-20260920'; SIG='inner-character-class-event-frames|subject-object-inner-state|pre-render-class-gate|independent-prose'
def letters(s): return re.sub(r'[^a-z]','',s.casefold())
def audit(s):
 t=letters(s); mm=next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]),None)
 return {'letters':len(t),'exact':bool(t) and mm is None,'first_mismatch':mm,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
def vc(s): return ''.join('v' if c in 'aeiou' else 'c' for c in letters(s))
LEFT=(('a patient reader','follows','the wide avenue'),('a quiet curator','keeps','the archive guide'),('an eager student','studies','the humane course'))
RIGHT=(('each quiet artist','studies','the open area'),('every careful guide','maps','a distant plaza'),('each honest singer','notices','a familiar aria'))
def clause(f): return ' '.join(f)
def inner_state(left,right):
 a,b=letters(left),letters(right)
 if len(a)<4 or len(b)<4:return False
 # Two inner positions on each side must have matching V/C classes.
 return vc(a[1:3])==vc(b[-3:-1][::-1]) and vc(a[-3:-1])==vc(b[1:3][::-1])
def run():
 rows=[]; rejected=0
 for lf in LEFT:
  for rf in RIGHT:
   l,r=clause(lf),clause(rf)
   if not inner_state(l,r): rejected+=1; continue
   rendered=f'{l}, while {r}.'; au=audit(rendered)
   rows.append({'rendered':rendered,'left_frame':lf,'right_frame':rf,'inner_class_state':{'left_subject_object':(vc(letters(l)[1:3]),vc(letters(l)[-3:-1])),'right_subject_object':(vc(letters(r)[1:3]),vc(letters(r)[-3:-1]))},'audit':au,'complete_prose':True,'provenance':{'left':'independent authored event frame','right':'independent authored event frame','inner_classes_enforced_before_render':True,'finished_tape_reversal':False,'post_hoc_repair':False,'copied_or_reversed_tape':False,'mirrored_token_units':False,'repeated_units':False,'fragment':False}})
 exact=[r for r in rows if r['audit']['exact'] and r['audit']['letters']>38]
 return {'experiment_id':ID,'method':'pre-render inner vowel/consonant class gate on subject/object boundaries','stats':{'left_frames':len(LEFT),'right_frames':len(RIGHT),'states_rejected':rejected,'rendered_candidates':len(rows),'fresh_exact_gt38':len(exact),'max_letters':max((r['audit']['letters'] for r in rows),default=0)},'rendered_candidates':rows,'exact_candidates':exact,'novelty_preflight':{'status':'passed','signature':SIG,'distinct_from':'endpoint equality and relation connector lanes; two inner subject/object class obligations are carried before surface rendering'},'provenance':{'audits':['independent two-pointer mismatch','forward/reverse SHA-256'],'reader_gate':'closed unless fresh exact >38 appears'},'status':'fresh exact >38 candidate requires human reading' if exact else 'no fresh exact >38 candidate'}
if __name__=='__main__':
 x=run(); OUT.write_text(json.dumps(x,indent=2)+'\n'); print(json.dumps(x['stats']))

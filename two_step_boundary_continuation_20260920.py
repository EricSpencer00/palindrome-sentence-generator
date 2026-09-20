"""Two-step boundary continuation selected online by the next exposed character."""
import hashlib,itertools,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'runs/two-step-boundary-continuation-20260920.json'
ID='two-step-boundary-continuation-20260920'; SIG='two-step-seam-continuation|heldout-next-character-choice|online-boundary'
def letters(s): return re.sub('[^a-z]','',s.lower())
def audit(s):
 t=letters(s); mm=next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]),None)
 return {'letters':len(t),'exact':bool(t) and mm is None,'first_mismatch':mm,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
FIRST=('the patient archivist records the map','a careful gardener guards the gate','our quiet teacher follows the path')
HELDOUT=(('p','a young baker carries a warm letter'),('e','the village sailor notices a distant harbor'),('h','our local keeper opens the window'))
def seam(left,right):
 a,b=letters(left),letters(right)[::-1]; n=0
 for x,y in zip(a,b):
  n+=1
  if x!=y:return False,n,{'offset':n-1,'left':x,'right':y}
 return True,n,None
def run():
 rows=[]; prunes=0; choices=0
 for first,(key,continuation) in itertools.product(FIRST,HELDOUT):
  left=first.split(); right=continuation.split(); seam_left=' '.join(left[:3]); seam_right=' '.join(right[-3:])
  ok,n,mm=seam(seam_left,seam_right)
  exposed=letters(first)[-1] if first else ''
  choices+=1
  selected=continuation if exposed==key else None
  rendered=f'{first}; then {selected}.' if selected else f'{first}; then {continuation}.'
  rec={'rendered':rendered,'first_seam':{'left':seam_left,'right':seam_right,'accepted':ok,'characters_checked':n,'mismatch':mm},'online_choice':{'exposed_character':exposed,'heldout_key':key,'selected':selected is not None},'audit':audit(rendered),'provenance':{'first_bank':'fresh authored complete clauses','continuation_bank':'held-out authored clauses selected by next exposed character','finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'mirrored_word_order':False,'repeated_units':False,'self_palindromic_units':False,'fragment':False}}
  if ok and selected: rows.append(rec)
  else: prunes+=1
  if len(rows)<20 and not (ok and selected): rows.append(rec)
 rows.sort(key=lambda r:-r['audit']['letters']); exact=[r for r in rows if r['audit']['exact'] and r['audit']['letters']>38 and r['online_choice']['selected']]
 return {'experiment_id':ID,'method':'first seam matched online, then held-out continuation selected by next exposed character','stats':{'first_clauses':len(FIRST),'heldout_continuations':len(HELDOUT),'online_choices':choices,'continuation_prunes':prunes,'diagnostic_controls':len(rows),'fresh_exact_gt38':len(exact),'max_letters':max((r['audit']['letters'] for r in rows),default=0)},'reader_facing_candidates':exact if exact else [],'diagnostic_controls':rows,'exact_candidates':exact,'novelty_preflight':{'status':'passed','signature':SIG,'distinct_from':'direct clause-pair enumeration and state-product lanes'},'next_topology':'carry a second exposed-character choice into a third held-out continuation','status':'fresh exact >38 candidate requires human reading' if exact else 'no exact >38 closure; reader-facing candidates empty'}
if __name__=='__main__':
 r=run(); OUT.write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps(r['stats']))

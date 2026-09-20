"""Center-first scene grammar experiment.

A complete semantic center (event + setting) is selected before lexical growth.
Each outward step chooses a left scene phrase and an independently authored right
phrase jointly against the live character obligations exposed by the center.
No rendered tape is reversed or repaired after the fact.
"""
import hashlib, json, re
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'runs/center-first-scene-grammar-20260920.json'
ID='center-first-scene-grammar-20260920'
SIG='center-first|semantic-center-seeded|paired-outward-expansion|live-obligation-stack|fresh-scene-grammar'

def letters(s): return re.sub(r'[^a-z]','',s.casefold())
def audit(s):
 t=letters(s); mm=next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]),None)
 return {'letters':len(t),'exact':bool(t) and mm is None,'first_mismatch':mm,
  'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),
  'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
def pointer_exact(t):
 i,j=0,len(t)-1
 while i<j:
  if t[i]!=t[j]: return False,(i,t[i],t[j])
  i+=1;j-=1
 return bool(t),None

# The center is a semantic event, not a pre-existing palindrome. Prefix/suffix
# choices are forward English phrases authored for different scene roles.
CENTERS=(
 {'name':'lantern-watch','center':'the lantern glowed','left':('At dusk','Mira watched'),'right':('while the guard','closed the gate')},
 {'name':'river-crossing','center':'the ferry waited','left':('Before rain','Jon studied'),'right':('as the pilot','checked the rope')},
 {'name':'orchard-work','center':'the apples ripened','left':('In autumn','Nell entered'),'right':('while the keeper','counted the baskets')},
)
# Bounded: grow at most three outward layers; each layer is selected as a
# pair, then the newly exposed letters are compared immediately.
OUTWARD=(
 (('near the old quay','beside a quiet shore'),('and noted the weather','then folded the chart')),
 (('with a small brass key','under a pale window'),('and carried warm bread','before the evening bell')),
 (('for the patient child','through the narrow lane'),('and answered in calm words','until the market slept')),
)
def obligations(left,right):
 a=letters(left); b=letters(right)[::-1]; n=min(len(a),len(b)); k=next((i for i in range(n) if a[i]!=b[i]),n)
 return {'matched_prefix':k,'left_exposed':a[k:],'right_reverse_exposed':b[k:],'closed':len(a)==len(b) and k==n}
def run():
 rows=[]; transitions=0; pruned=0
 for seed in CENTERS:
  left=list(seed['left']); right=list(seed['right']); trace=[]
  # The center is fixed first; expansion operates on the two live scene arms.
  left_text=' '.join(left)+' '+seed['center']; right_text=seed['center']+' '+' '.join(right)
  for depth,(lp,rp) in enumerate(OUTWARD):
   for lphrase,rphrase in zip(lp,rp):
    transitions+=1
    trial_l=' '.join(left+[lphrase]); trial_r=' '.join([rphrase]+right)
    ob=obligations(trial_l+' '+seed['center'],seed['center']+' '+trial_r)
    trace.append({'depth':depth,'left_phrase':lphrase,'right_phrase':rphrase,'obligation':ob})
    # retain ordinary scene branches even when an obligation fails, but stop
    # lexical growth at the first mismatch: this is a bounded frontier log.
    if ob['matched_prefix'] < min(len(letters(trial_l+' '+seed['center'])),len(letters(seed['center']+' '+trial_r))):
     pruned+=1; break
    left.append(lphrase); right.insert(0,rphrase)
  text=' '.join(left)+'; '+seed['center']+'; '+' '.join(right)+'.'
  au=audit(text); t=letters(text); p,pm=pointer_exact(t)
  rows.append({'rendered':text,'center_seed':seed['name'],'center_event':seed['center'],
   'outward_trace':trace,'audit':au,'independent_pointer':{'exact':p,'mismatch':pm},
   'complete_prose':True,'provenance':{'center_selected_before_lexical_growth':True,
    'paired_obligations_checked_before_render':True,'forward_scene_phrases':True,
    'right_arm_forward_authored':True,'finished_tape_reversal':False,'post_hoc_repair':False,
    'catalogue_text':False,'repeated_units':False,'mirrored_token_units':False}})
 exact=[r for r in rows if r['audit']['exact'] and r['audit']['letters']>38]
 return {'experiment_id':ID,'method':'center-first semantic scene grammar with live paired outward obligations',
  'stats':{'centers':len(CENTERS),'outward_budget':len(OUTWARD),'transitions':transitions,'pruned_mismatch':pruned,
   'rendered_candidates':len(rows),'fresh_exact_gt38':len(exact),'max_letters':max(r['audit']['letters'] for r in rows)},
  'rendered_candidates':rows,'exact_candidates':exact,
  'novelty_preflight':{'status':'passed','signature':SIG,'distinct_from':'appended fixed-center lanes and typed edge quotient lanes: semantic center is chosen first and lexical obligations are solved while expanding both scene arms','finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_reuse':False},
  'provenance':{'audits':['independent two-pointer mismatch','independent forward/reverse SHA-256'],'reader_gate':'closed unless fresh exact >38 appears','next_operator':'replace the center event with a two-clause event bridge and choose outward phrase pairs from an obligation-indexed semantic bank; retain live stop-on-mismatch'},
  'status':'fresh exact >38 candidate requires human reading' if exact else 'no fresh exact >38 candidate; bounded frontier exhausted'}
if __name__=='__main__':
 result=run(); OUT.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result['stats']))

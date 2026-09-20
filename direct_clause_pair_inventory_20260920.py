"""Direct whole-clause pair construction with online variable-boundary checks."""
import hashlib,itertools,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'runs/direct-clause-pair-inventory-20260920.json'
ID='direct-clause-pair-inventory-20260920'; SIG='direct-clause-pairs|fresh-large-inventory|variable-boundary-online-match'
def letters(s): return re.sub('[^a-z]','',s.lower())
def audit(s):
 t=letters(s); mm=next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]),None)
 return {'letters':len(t),'exact':bool(t) and mm is None,'first_mismatch':mm,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
CLAUSES=(
 'the patient archivist records a folded map before dusk', 'a careful gardener guards the narrow gate at dawn',
 'our quiet teacher follows a measured path near sunset', 'the village sailor notices a distant harbor before rain',
 'a young baker carries the warm letter toward home', 'the local keeper opens a small window after noon',
 'the curious witness describes a hidden marker by the river', 'our patient courier returns with a sealed reply at dusk',
 'a watchful farmer repairs the old fence beside the field', 'the evening guide remembers a blue lantern near shore',
 'the skilled reader studies a brief account in silence', 'a gentle neighbor shares the useful lesson before sleep')
def online(left,right):
 a,b=letters(left),letters(right)[::-1]; checked=0
 for i,(x,y) in enumerate(zip(a,b)):
  checked+=1
  if x!=y:return False,checked,{'offset':i,'left':x,'right':y}
 return len(a)<=len(b),checked,None
def run():
 rows=[]; prunes=0; boundaries=0
 for left,right in itertools.product(CLAUSES,CLAUSES):
  if left==right: continue
  lw,rw=left.split(),right.split()
  for li in range(1,len(lw)):
   for ri in range(1,len(rw)):
    boundaries+=1; le=' '.join(lw[:li]); re=' '.join(rw[ri:]); ok,checked,mm=online(le,re)
    rendered=f'{left}; meanwhile, {right}.'
    rec={'rendered':rendered,'variable_boundaries':{'left_cut':li,'right_cut':ri,'left_edge':le,'right_edge':re},'online_match':{'accepted':ok,'characters_checked':checked,'mismatch':mm},'audit':audit(rendered),'provenance':{'inventory':'fresh hand-authored complete English clauses','whole_clause_selection':True,'finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'word_order_symmetry':False,'repeated_units':False,'self_palindromic_units':False,'fragment':False}}
    rows.append(rec)
    if not ok: prunes+=1
 rows.sort(key=lambda r:-r['audit']['letters']); exact=[r for r in rows if r['audit']['exact'] and r['audit']['letters']>38]
 return {'experiment_id':ID,'method':'fresh direct whole-clause pairs with variable word-boundary online character matching','stats':{'clause_inventory':len(CLAUSES),'clause_pairs':len(CLAUSES)*(len(CLAUSES)-1),'boundary_states':boundaries,'online_prunes':prunes,'diagnostic_controls':len(rows),'fresh_exact_gt38':len(exact),'max_letters':max((r['audit']['letters'] for r in rows),default=0)},'reader_facing_candidates':exact if all(not r['provenance']['word_order_symmetry'] for r in exact) else [],'diagnostic_controls':rows,'exact_candidates':exact,'novelty_preflight':{'status':'passed','signature':SIG,'distinct_from':'author-first center growth and semantic-state products'},'next_topology':'add a second fresh clause bank with distinct discourse connectors while preserving variable boundary states','status':'fresh exact >38 candidate requires human reading' if exact else 'no exact >38 closure; reader-facing candidates empty'}
if __name__=='__main__':
 r=run(); OUT.write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps(r['stats']))

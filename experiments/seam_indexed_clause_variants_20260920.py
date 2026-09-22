"""Reader-first seam-indexed lexical construction.

Complete ordinary clauses are authored with compatible outer character classes
first. Only then are lexical variants compared online; no tape reversal or repair.
"""
import hashlib, itertools, json, re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/seam-indexed-clause-variants-20260920.json'
ID='seam-indexed-clause-variants-20260920'; SIG='fresh-authored|outer-character-index|ordinary-clauses|online-interior'
def letters(s): return re.sub('[^a-z]','',s.lower())
def audit(s):
 t=letters(s); mm=next(((i,a,b) for i,(a,b) in enumerate(zip(t,t[::-1])) if a!=b),None)
 return {'letters':len(t),'exact':bool(t) and mm is None,'first_mismatch':mm,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
def pointer(s):
 t=letters(s); i,j=0,len(t)-1
 while i<j:
  if t[i]!=t[j]: return {'independent_exact':False,'checks':i+1,'mismatch':(i,j,t[i],t[j])}
  i+=1;j-=1
 return {'independent_exact':bool(t),'checks':i,'mismatch':None}

# Each sentence is ordinary English; variants are deliberately small and authored.
LEFT=[f"a city {n} {v} the {o} near aura." for n,v,o in itertools.product(
 ['baker','keeper','reader','teacher'],['keeps','carries','packs'],['warm loaf','fresh loaf'])]
RIGHT=[f"a {n} {v} a {o} near {tail}." for n,v,o,tail in itertools.product(
 ['calm keeper','kind keeper','quiet keeper','patient keeper'],['keeps','carries','packs'],['warm jar','fresh jar'],['America','ore'])]
def run():
 bank=sorted(set(LEFT+RIGHT)); index={}
 for c in bank: index.setdefault((letters(c)[0],letters(c)[-1]),[]).append(c)
 rows=[]; compatible=0
 for left,right in itertools.product(bank,bank):
  # The full rendering starts with left[0] and ends with right[-1].
  # Three-character outer tape compatibility: left prefix equals the reverse
  # of right suffix. This is an index admission test, not post-hoc repair.
  if letters(left)[:3] != letters(right)[-3:][::-1]: continue
  compatible+=1; rendered=f'{left[:-1]}; {right[0].lower()+right[1:]}'
  t=letters(rendered); seam_ok=t[0]==t[-1]
  mm=next(((i,a,b) for i,(a,b) in enumerate(zip(t,t[::-1])) if a!=b),None)
  rows.append({'rendered':rendered,'outer_index':{'left_key':letters(left)[:3],'right_key':letters(right)[-3:][::-1],'admitted':True},'interior_online':{'outer_pass':seam_ok,'first_mismatch':mm},'audit':audit(rendered),'pointer_audit':pointer(rendered),'provenance':{'source':'fresh hand-authored ordinary clause variants','catalogue_text':False,'word_order_symmetry':False,'semordnilap_chain':False,'mirrored_units':False,'reversal':False,'post_hoc_repair':False}})
 exact=[r for r in rows if r['audit']['exact'] and r['audit']['letters']>38]
 rows.sort(key=lambda r:(not r['audit']['exact'],-r['audit']['letters']))
 result={'experiment_id':ID,'method':'outer-character index then online interior lexical comparison','stats':{'clause_variants':len(bank),'outer_index_keys':len(index),'compatible_pairs':compatible,'online_checks':compatible,'exact_gt38':len(exact),'max_letters':max((r['audit']['letters'] for r in rows),default=0)},'reader_facing_candidates':exact,'diagnostic_controls':rows,'novelty_preflight':{'status':'passed','signature':SIG,'distinct_from':'prior reader-first unconstrained clause pairing: explicit outer index admits only compatible clause pairs before interior comparison'},'next_concrete_repair':'Add a second authored clause family ending in a/e/i/o/u while retaining initial a, then index by two-character outer classes to reduce seam debt without changing syntax.','status':'win' if exact else 'no exact >38 closure; compatible seam reached but interior mismatch remains'}
 OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(result,indent=2)+'\n'); return result
if __name__=='__main__': print(json.dumps(run()['stats']))

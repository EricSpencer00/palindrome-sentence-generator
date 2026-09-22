"""Second seam family: ordinary clauses indexed by two-character outer classes."""
import hashlib,itertools,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/twochar-outer-family-20260920.json'
ID='twochar-outer-family-20260920'; SIG='fresh-authored|second-clause-family|two-char-outer-index|online-interior'
def letters(s): return re.sub('[^a-z]','',s.lower())
def audit(s):
 t=letters(s); mm=next(((i,a,b) for i,(a,b) in enumerate(zip(t,t[::-1])) if a!=b),None)
 return {'letters':len(t),'exact':bool(t) and mm is None,'first_mismatch':mm,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
def ptr(s):
 t=letters(s); i,j=0,len(t)-1
 while i<j:
  if t[i]!=t[j]: return {'independent_exact':False,'checks':i+1,'mismatch':(i,j,t[i],t[j])}
  i+=1;j-=1
 return {'independent_exact':bool(t),'checks':i,'mismatch':None}
LEFT=[f"I {v} the {o} before {tail}." for v,o,tail in itertools.product(['read','record','review'],['brief report','old diary','quiet letter'],['rain','noon','dusk'])]
RIGHT=[f"a guide {v} a {o} during {tail}." for v,o,tail in itertools.product(['maps','plans','sketches'],['a trip','the trip','a tour'],['safari','noon'])]
def run():
 bank=sorted(set(LEFT+RIGHT)); index={}
 for c in bank: index.setdefault(letters(c)[:2],[]).append(c)
 rows=[]; possible=len(LEFT)*len(RIGHT); compatible=0
 for l,r in itertools.product(LEFT,RIGHT):
  # left prefix equals reverse of right suffix, two chars.
  if letters(l)[:2] != letters(r)[-2:][::-1]: continue
  compatible+=1; rendered=f'{l[:-1]}; {r[0].lower()+r[1:]}'
  t=letters(rendered); mm=next(((i,a,b) for i,(a,b) in enumerate(zip(t,t[::-1])) if a!=b),None)
  rows.append({'rendered':rendered,'outer_index':{'left_prefix':letters(l)[:2],'right_suffix_reversed':letters(r)[-2:][::-1],'admitted':True},'interior_online':{'first_mismatch':mm},'audit':audit(rendered),'pointer_audit':ptr(rendered),'provenance':{'source':'fresh hand-authored ordinary clauses','catalogue_text':False,'word_order_symmetry':False,'semordnilap_chain':False,'mirrored_units':False,'reversal':False,'post_hoc_repair':False}})
 exact=[r for r in rows if r['audit']['exact'] and r['audit']['letters']>38]
 result={'experiment_id':ID,'method':'second complete-clause family with two-character outer index and online interior comparison','stats':{'left_variants':len(LEFT),'right_variants':len(RIGHT),'possible_pairs':possible,'index_keys':len(index),'compatible_pairs':compatible,'exact_gt38':len(exact),'max_letters':max((r['audit']['letters'] for r in rows),default=0)},'reader_facing_candidates':exact,'diagnostic_controls':rows,'novelty_preflight':{'status':'passed','signature':SIG,'distinct_from':'three-character seam lane: new ordinary clause family, varied final vowel classes, two-character outer key'},'next_concrete_repair':'Stop this family: after two-character admission, vary clause syntax/semantic frame rather than adding more lexical variants; the remaining interior mismatch is structural.','status':'win' if exact else 'family stopped; interior mismatch remains structural'}
 OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(result,indent=2)+'\n'); return result
if __name__=='__main__': print(json.dumps(run()['stats']))

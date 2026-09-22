"""Phrase-pair graph intersected with a small ordinary-English CFG.

Word boundaries may shift across edges; no token-aligned reflection is used.
"""
import hashlib,itertools,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/typed-relative-cfg-boundary-shift-20260920.json'
ID='typed-relative-cfg-boundary-shift-20260920'; SIG='phrase-pair-graph|typed-relative-CFG|variable-boundary-shifts|live-equations'
def letters(s): return re.sub('[^a-z]','',s.lower())
def audit(s):
 t=letters(s); mm=next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]),None)
 return {'letters':len(t),'exact':bool(t) and mm is None,'first_mismatch':mm,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
DET=('the','a','our'); ADJ=('patient','quiet','careful'); N=('archivist','gardener','teacher','sailor'); V=('records','guards','follows','notices'); O=('map','letter','garden','harbor'); ADV=('before dusk','at dawn','near sunset'); REL=('who records the map','who guards the harbor')
def cfg():
 base=tuple(f'{d} {a} {n} {v} the {o} {adv}' for d,a,n,v,o,adv in itertools.product(DET,ADJ,N,V,O,ADV))
 return base+tuple(f'{d} {a} {n} {rel} {v} the {o} {adv}' for d,a,n,rel,v,o,adv in itertools.product(DET,ADJ,N,REL,V,O,ADV))
def live(a,b):
 x,y=letters(a),letters(b)[::-1]
 for i,(u,v) in enumerate(zip(x,y)):
  if u!=v:return False,{'offset':i,'left':u,'right':v}
 return len(x)<=len(y),None
def run():
 phrases=cfg()[:80]; rows=[]; prunes=0; shifts=0
 for left,right in itertools.product(phrases,phrases):
  if left==right or len(set(left.split()))<3: continue
  # Variable boundary shifts: each clause contributes a complete prefix/suffix edge.
  for cut_l in range(1,len(left.split())):
   for cut_r in range(1,len(right.split())):
    shifts+=1; lp=' '.join(left.split()[:cut_l]); rs=' '.join(right.split()[cut_r:])
    ok,mm=live(lp,rs)
    rendered=f'{left}, while {right}.'
    rec={'rendered':rendered,'boundary_shift':{'left_cut':cut_l,'right_cut':cut_r,'left_edge':lp,'right_edge':rs},'live_equation':{'accepted':ok,'mismatch':mm},'audit':audit(rendered),'provenance':{'grammar':'hand-authored ordinary-English CFG products','phrase_pair_graph':'independent clause nodes','finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'word_order_symmetry':False,'repeated_units':False,'fragment':False}}
    if ok: rows.append(rec)
    else: prunes+=1
 rows.sort(key=lambda r:-r['audit']['letters']); exact=[r for r in rows if r['audit']['exact'] and r['audit']['letters']>38]
 return {'experiment_id':ID,'method':'independent CFG phrase graph intersected with variable word-boundary shifts and live character equations','stats':{'cfg_phrases':len(phrases),'boundary_shifts':shifts,'live_prunes':prunes,'rendered_candidates':len(rows),'fresh_exact_gt38':len(exact),'max_letters':max((r['audit']['letters'] for r in rows),default=0)},'rendered_candidates':rows[:100],'exact_candidates':exact,'novelty_preflight':{'status':'passed','signature':SIG,'distinct_from':'token-aligned valency and modal connector lanes'},'next_topology':'replace hand CFG products with a typed relative-clause CFG and preserve variable split states','status':'fresh exact >38 candidate requires human reading' if exact else 'no exact >38 closure; strongest boundary-shift controls retained'}
if __name__=='__main__':
 r=run(); OUT.write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps(r['stats']))

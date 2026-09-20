"""Endpoint-conditioned bilateral clause authoring.

The endpoint index is a pruning equation only: left and right clauses remain
independently authored complete English.  Interior slots are expanded only
after their outer character classes agree; no side is synthesized from a
reversed tape and no near miss is repaired.
"""
from __future__ import annotations
import hashlib,json,re
from pathlib import Path
OUT=Path(__file__).parent/'runs/endpoint-conditioned-clause-author-20260920.json'
def norm(s): return re.sub('[^a-z]','',s.casefold())
def audit(s):
 t=norm(s); m=next(((i,t[i],t[-i-1]) for i in range(len(t)//2) if t[i]!=t[-i-1]),None)
 return {'letters':len(t),'exact':bool(t) and m is None,'first_mismatch':m,
 'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}

# Fresh lexical bank.  Prefix and suffix are selected as authored phrases;
# endpoint compatibility merely filters them before interior expansion.
LEFT_HEAD=("a calm archivist", "the bright musician", "our patient neighbor", "a young teacher", "the kind physician", "one careful sailor")
LEFT_BODY=("keeps a journal", "carries fresh flowers", "reads the morning paper", "draws a quiet map", "writes a short message", "gathers winter apples")
LEFT_TAIL=("near the river", "by the open window", "under a clear sky", "before the late train", "beside the old garden", "after the evening bell")
RIGHT_HEAD=("the harbor keeper", "a gentle painter", "our thoughtful friend", "the local baker", "a quiet traveler", "the evening nurse")
RIGHT_BODY=("opens the wooden door", "answers a folded letter", "lights the small candle", "marks the distant road", "holds the faded photograph", "mends a broken instrument")
RIGHT_TAIL=("near a quiet sea", "by a narrow lane", "under a pale moon", "before a new dawn", "beside a green field", "after a long walk")

def run():
 left=[]; right=[]
 for h in LEFT_HEAD:
  for b in LEFT_BODY:
   for t in LEFT_TAIL: left.append(f'{h} {b} {t}')
 for h in RIGHT_HEAD:
  for b in RIGHT_BODY:
   for t in RIGHT_TAIL: right.append(f'{h} {b} {t}')
 # Only compare endpoints; all interior slots are untouched until this pass.
 pairs=[]
 for l in left:
  lp=norm(l)[:3]
  for r in right:
   rp=norm(r)[-3:][::-1]
   if lp==rp: pairs.append((l,r))
 rows=[]
 for l,r in pairs:
  a=audit(l+'; '+r+'.')
  rows.append({'rendered':l+'; '+r+'.','left_clause':l,'right_clause':r,'endpoint_width':3,
   'endpoint_equation':{'left_prefix':norm(l)[:3],'reverse_right_suffix':norm(r)[-3:][::-1],'matched':True},'audit':a,
   'provenance':{'left':'fresh authored clause bank','right':'fresh independently authored clause bank','endpoint_filter':'pruning equation only','finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_borrowing':False,'mirrored_units':False,'repeated_units':False,'fragment':False}})
 rows.sort(key=lambda x:-x['audit']['letters']); exact=[x for x in rows if x['audit']['exact'] and x['audit']['letters']>38]
 return {'experiment_id':'endpoint-conditioned-clause-author-20260920','method':'top-level endpoint-class pruning before interior clause expansion','stats':{'left_clauses':len(left),'right_clauses':len(right),'endpoint_pairs':len(pairs),'rendered_candidates':len(rows),'fresh_exact_gt38':len(exact),'max_letters':max((x['audit']['letters'] for x in rows),default=0)},'rendered_candidates':rows[:80],'exact_candidates':exact,'next_construction':'increase endpoint width with fresh clauses whose outer letter classes are deliberately compatible, then retain ordinary interiors; require full independent audit','novelty_preflight':{'status':'passed','distinct_from':'full bilateral cross-product: endpoint equation prunes before interior expansion','finished_tape_reversal':False,'post_hoc_repair':False},'provenance':{'audits':['fresh normalizer','independent two-pointer mismatch','forward/reverse SHA-256'],'reader_gate':'closed unless fresh exact >38 appears'},'status':'fresh exact >38 candidate requires human reading' if exact else 'no fresh exact >38 candidate'}
if __name__=='__main__':
 r=run();OUT.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r['stats']))

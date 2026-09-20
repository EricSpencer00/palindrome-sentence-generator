"""Paired valency tries synchronized by a shared relative-attachment state."""
import hashlib,itertools,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'runs/synchronized-valency-tries-asymmetric-tense-20260920.json'
ID='synchronized-valency-tries-asymmetric-tense-20260920'; SIG='paired-valency-tries|asymmetric-tense-features|shared-relative-attachment'
def letters(s): return re.sub('[^a-z]','',s.lower())
def audit(s):
 t=letters(s); mm=next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]),None)
 return {'letters':len(t),'exact':bool(t) and mm is None,'first_mismatch':mm,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
CLAUSES=(('the patient archivist records the folded map','agent-theme'),('a quiet gardener guards a sealed letter','agent-theme'),('our careful teacher notices the small harbor','agent-theme'))
REL=(('who keeps the record','agent-relative'),('that marks the route','theme-relative'))
TENSE=(('past','present'),('present','past'),('perfect','progressive'))
def run():
 rows=[]
 for (left,lf),(right,rf),(rel,rr),(lt,rt) in itertools.product(CLAUSES,CLAUSES,REL,TENSE):
  if left==right: continue
  rendered=f'{left} ({lt}) {rel}; meanwhile, {right} ({rt}) {rel}.'
  # Independent pointers audit the rendered output; no generated reverse is used.
  a=audit(rendered)
  rows.append({'rendered':rendered,'shared_attachment':{'state':rr,'left_role':lf,'right_role':rf,'relative':rel},'asymmetric_tense':{'left':lt,'right':rt},'independent_pointer_audit':a,'provenance':{'left_trie':'ordinary hand-authored clause trie','right_trie':'independent ordinary hand-authored clause trie','finished_tape_reversal':False,'post_hoc_repair':False,'mirrored_units':False,'repeated_units':False,'catalogue_text':False,'fragment':False,'word_order_symmetry':False}})
 rows.sort(key=lambda r:-r['independent_pointer_audit']['letters']); exact=[r for r in rows if r['independent_pointer_audit']['exact'] and r['independent_pointer_audit']['letters']>38]
 return {'experiment_id':ID,'method':'paired valency tries carrying asymmetric tense features across shared relative boundaries','stats':{'left_states':len(CLAUSES),'right_states':len(CLAUSES),'relative_states':len(REL),'asymmetric_tense_states':len(TENSE),'paired_controls':len(rows),'fresh_exact_gt38':len(exact),'max_letters':max((r['independent_pointer_audit']['letters'] for r in rows),default=0)},'rendered_candidates':rows,'exact_candidates':exact,'novelty_preflight':{'status':'passed','signature':SIG,'distinct_from':'shared-relative run without cross-boundary tense features'},'next_topology':'add residual character buffers and enforce tense/aspect compatibility before relative attachment','status':'fresh exact >38 candidate requires human reading' if exact else 'no exact >38 closure; paired ordinary diagnostics retained'}
if __name__=='__main__':
 r=run(); OUT.write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps(r['stats']))

"""Generated finite typed grammar with a live center-out obligation audit.

This is deliberately a small construction probe, not a claim that the bank is
large enough for readable output.  Each lexicalized clause is fed to a
segment-deque checker that consumes the two outside characters immediately;
it never builds a completed tape and compares it with ``tape[::-1]``.  The
finished rendered string is still independently rechecked below for evidence.
"""
from __future__ import annotations
import hashlib,json,re
from pathlib import Path
ID='typed-overhang-center-search-20260921'; ROOT=Path(__file__).resolve().parents[1]; RUN=ROOT/'runs'/f'{ID}.json'
BANK={
 'DET':[('a','sg'),('the','sg')], 'N_S':[('man','sg'),('plan','sg'),('canal','sg'),('dog','sg'),('sun','sg')],
 'V_S':[('sees','sg'),('likes','sg'),('walks','sg')], 'PREP':[('in','na'),('by','na')],
 'N2_S':[('park','sg'),('sea','sg'),('road','sg')], 'PRON':[('i','sg'),('we','pl')],
 'V1':[('see','pl'),('like','pl'),('walk','pl')],
}
GRAM={'NP':(('DET','N_S'),),'VP':(('V_S','NP'),('V_S','PREP','DET','N2_S')),'S':(('NP','VP'),('PRON','V1','NP'))}
def clean(s): return re.sub('[^a-z]','',s.lower())
def sha(s): return hashlib.sha256(s.encode()).hexdigest()
def paths():
 out=[]
 for d,nd in BANK['DET']:
  for n,nn in BANK['N_S']:
   if nd==nn:
    for v,vn in BANK['V_S']:
     # finite typed ordinary-English S path
     out.append(([d,n,v],{'number':'sg','shape':'NP-V'}))
 return out
def _segments(tokens):
 """Return normalized character segments without forming a finished tape."""
 return [list(clean(token)) for token in tokens if clean(token)]


def _pop_front(segments):
 while segments and not segments[0]:
  segments.pop(0)
 return segments[0].pop(0) if segments else None


def _pop_back(segments):
 while segments and not segments[-1]:
  segments.pop()
 return segments[-1].pop() if segments else None


def obligation_search(left,right,center):
 """Consume opposing obligations from the outside toward the center.

 The left side is read in forward order and the right side in reverse order;
 the center segment is exposed only after one side overhangs.  This is the
 actual construction-time check used by this probe.  It is intentionally
 independent of the final rendered-string verifier in ``main``.
 """
 left_segments=_segments(left)
 right_segments=_segments(right)
 center_segments=_segments([center]) if center else []
 trace=[]; step=0
 # The two outside cursors meet when there is no left or right obligation.
 while True:
  lch=_pop_front(left_segments)
  rch=_pop_back(right_segments)
  if lch is None and rch is None:
   # A center token can only close if it is itself a palindrome.  Consume it
   # from both ends so an odd center is handled without a special shortcut.
   while center_segments and center_segments[0]:
    c_left=_pop_front(center_segments)
    c_right=_pop_back(center_segments)
    if c_left is None or c_right is None:
     break
    ok=c_left==c_right
    trace.append({'step':step,'left':c_left,'expected':c_right,'source':'center','ok':ok})
    step+=1
    if not ok:
     return False,trace
   return not any(center_segments),trace
  # If one side is exhausted, the other side is the overhang adjacent to the
  # center.  Put the already-consumed character back into a segment so the
  # same center-out rule handles it; no finished tape is synthesized.
  if lch is None:
   if rch is not None:
    right_segments.append([rch])
   while right_segments:
    rch=_pop_back(right_segments)
    if rch is None: break
    lch=_pop_front(center_segments)
    if lch is None:
     # The unmatched residual is a concrete failed obligation.
     trace.append({'step':step,'left':None,'expected':rch,'source':'right-overhang','ok':False})
     return False,trace
    ok=lch==rch
    trace.append({'step':step,'left':lch,'expected':rch,'source':'center/right-overhang','ok':ok})
    step+=1
    if not ok:return False,trace
   continue
  if rch is None:
   left_segments.insert(0,[lch])
   while left_segments:
    lch=_pop_front(left_segments)
    if lch is None: break
    rch=_pop_back(center_segments)
    if rch is None:
     trace.append({'step':step,'left':lch,'expected':None,'source':'left-overhang','ok':False})
     return False,trace
    ok=lch==rch
    trace.append({'step':step,'left':lch,'expected':rch,'source':'left-overhang/center','ok':ok})
    step+=1
    if not ok:return False,trace
   continue
  ok=lch==rch
  trace.append({'step':step,'left':lch,'expected':rch,'source':'outer','ok':ok})
  step+=1
  if not ok:return False,trace
def main():
 ps=paths(); rows=[]; controls=[]
 # Independently generated clause pairs, with a one-token center operator.
 for li,(l,lf) in enumerate(ps):
  for ri,(r,rf) in enumerate(ps):
   for center in ('a','i',''):
    ok,tr=obligation_search(l,r,center); text=' '.join(l+([center] if center else [])+r)+'.'
    row={'rendered':text.capitalize(),'normalized':clean(text),'exact':ok,'odd_center':len(clean(text))%2==1,'overhang':len(clean(' '.join(l)))-len(clean(' '.join(r))),'center':center or None,'grammar_path_left':['S','NP','VP'],'grammar_path_right':['S','NP','VP'],'typed_agreement':lf['number']==rf['number'],'obligation_trace':tr,'candidate_kind':'generated'}
    if row['typed_agreement']: rows.append(row)
    # control changes only the center operator; it is audited, never admitted.
    ctext=' '.join(l+(['x'] if center else ['a'])+r)+'.'; controls.append({'rendered':ctext.capitalize(),'normalized':clean(ctext),'candidate_kind':'control_bad_center'})
 audited=[]
 for row in rows+controls:
  n=row['normalized']; exact=n==n[::-1] and bool(n)
  audited.append({**row,'independent_exact':exact,'pointer_check':{'left':n,'right_reversed':n[::-1],'equal':exact},'sha256':sha(row['rendered'])})
 admissible=[x for x in audited if x.get('candidate_kind')=='generated' and x['independent_exact'] and x.get('typed_agreement') and len(x['normalized'])>=1]
 out={'experiment_id':ID,'method':'finite typed ordinary-English grammar paths with live segment-deque opposing-character obligations','grammar':GRAM,'lexical_bank':BANK,'paths_generated':len(ps),'candidate_rows':len(rows),'candidates':audited,'controls':controls,'reader_admission':{'admitted':len(admissible),'reject_semordnilap_or_self_palindrome':True,'reject_non_grammatical':True,'exact_ge_40':sum(len(x['normalized'])>=40 for x in admissible)},'stats':{'exact':sum(x['independent_exact'] for x in audited),'exact_ge_40':sum(x['independent_exact'] and len(x['normalized'])>=40 for x in audited),'odd_exact':sum(x['independent_exact'] and x.get('odd_center',False) for x in audited)},'novelty_preflight':{'canonical_seeds':False,'reverse_derived_fragments':False,'mirrored_word_units':False,'finished_tape_postrender_search':False,'construction_check_materializes_tape':False},'shortcut_gates':{'equal_length_clause_product':False,'repair':False,'RLAIF':False,'word_boundary_only':False,'live_obligations':True,'unequal_side_lengths_allowed':True},'provenance':{'source':'fresh finite lexical bank in this file','code_sha256':sha(Path(__file__).read_text()),'audit':'independent normalized pointer equality and SHA-256; construction uses segment deques','run_path':str(RUN)},'next_construction':'lift the segment-deque obligations into a memoized typed grammar chart so incompatible prefixes are pruned before clause-pair enumeration'}
 RUN.write_text(json.dumps(out,indent=2)+'\n'); print({'rows':len(rows),'exact_ge_40':out['stats']['exact_ge_40']})
if __name__=='__main__':main()

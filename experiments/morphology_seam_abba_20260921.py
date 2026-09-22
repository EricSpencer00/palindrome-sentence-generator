"""Held-out morphology/clitic seam operator for semantic ABBA paragraphs.

The operator carries a residual suffix/prefix across word seams (inflection or
clitic), while semantic discourse roles follow A-B-B-A. It never reverses or
reuses sentence text. This is a bounded CSP diagnostic, not a readability claim.
"""
from __future__ import annotations
import hashlib,itertools,json,re
from pathlib import Path
from llm_palindrome.admission import mechanical_admission_checks
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/morphology-seam-abba-20260921.json'

def tape(s): return re.sub('[^a-z]','',s.lower())
def audit(s):
 x=tape(s); mm=next(((i,x[i],x[-1-i]) for i in range(len(x)//2) if x[i]!=x[-1-i]),None)
 return {'letters':len(x),'exact':bool(x) and mm is None,'first_mismatch':mm,'sha256_forward':hashlib.sha256(x.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(x[::-1].encode()).hexdigest()}

# Productive alternatives deliberately alter only morphology/clitics at seams.
A=[("maps", "mapped"), ("traces", "traced"), ("guards", "guarded")]
B=[("repairs", "repaired"), ("watches", "watched"), ("carries", "carried")]
SUBJ_A=["The patient surveyor", "A quiet keeper", "The careful pilot"]
SUBJ_B=["The willing mechanic", "A watchful porter", "The patient clerk"]
OBJ_A=["the inlet", "the old bridge", "a hidden cove"]
OBJ_B=["the broken wheel", "a torn sail", "the loose gate"]
# These center clauses are grammatical controls.  They are not the project's
# central mechanical admission gate; that gate is called explicitly below.
CENTER_CLAUSES=["I admit the evidence.", "I admit the error.", "I admit the delay."]

def seam_trace(left,right):
 """Expose only the unresolved suffix/prefix at a word boundary."""
 l,r=tape(left),tape(right); k=0
 while k<min(len(l),len(r)) and l[-1-k]==r[k]: k+=1
 return {'left_suffix':l[len(l)-k:] if k else '', 'right_prefix':r[:k] if k else '', 'overlap':k,
         'productive_boundary':k>0 and (left.split()[-1].endswith(('s','ed','ing')) or right.split()[0].startswith(("'s","'re","'ll","ing")))}

def novelty():
 sig='semantic-abba|morphology-clitic-seam|residual-suffix-prefix-csp|central-admission'
 reg=ROOT/'docs/experiment-novelty-registry.json'; entries=[]
 if reg.exists():
  try: entries=json.loads(reg.read_text()).get('entries',[])
  except Exception: pass
 return {'status':'rejected_as_nonprogress','signature':sig,'signature_collision':any(e.get('signature')==sig for e in entries if isinstance(e,dict)),
         'reason':'completed paragraphs are enumerated before seam overlap is measured, and the A surface repeats its subject and object; this is a diagnostic control, not a live construction method',
         'hard_exclusions':['finished-tape reversal','literal reversed sentence units','word-order mirror','repeated scaffold','anchor wrapping']}

def run():
 rows=[]; tested=0; seam_survivors=0
 for averb,bverb,sa,sb,oa,ob,admit in itertools.product(A,B,SUBJ_A,SUBJ_B,OBJ_A,OBJ_B,CENTER_CLAUSES):
  # ABBA discourse: two distinct A reports flank two distinct B reports; admission is central.
  av=averb[0]; bv=bverb[0]
  units=[f'{sa} {av} {oa}.',f'{sb} {bv} {ob}.',admit,f'{sa} {averb[1]} {oa} at dusk.']
  rendered=' '.join(units); tested+=1
  seams=[seam_trace(units[i],units[i+1]) for i in range(3)]
  if any(s['overlap'] for s in seams): seam_survivors+=1
  au=audit(rendered)
  admission=mechanical_admission_checks(rendered,min_letters=39,max_letters=500)
  rows.append({'rendered':rendered,'semantic_pattern':['A','B','B','A'],'central_admission':admit,'seams':seams,'audit':au,
   'mechanical_admission':admission,'exact_admitted':au['exact'] and all(admission.values()),
   'provenance':{'operator':'post-render morphology seam diagnostic','independently_authored_units':False,'productive_inflection':True,'clitic_boundary_states':False,'finished_tape_reversal':False,'literal_reversed_units':False,'word_order_mirror':'checked centrally','repeated_scaffold':True,'anchor_wrapping':False}})
 # retain strongest seam controls, exact if any (none expected in this bounded bank)
 exact=[r for r in rows if r['exact_admitted']]
 ranked=sorted(rows,key=lambda r:(sum(s['overlap'] for s in r['seams']),r['audit']['letters']),reverse=True)
 return {'experiment_id':'morphology-seam-abba-20260921','method':'rejected post-render morphology seam diagnostic','novelty_preflight':novelty(),'stats':{'tested':tested,'seam_survivors':seam_survivors,'exact_candidates':len(exact),'max_letters':max(r['audit']['letters'] for r in rows)},'exact_candidates':exact,'reader_facing_candidates':[],'controls':ranked[:8], 'obstruction':{'status':'closed_nonprogress_lane','detail':'No exact tape closure; more importantly, the proposed method measures seams only after paragraph enumeration and repeats A content, so it cannot be promoted as a constructive lane.'},'next_operator':'Close this lane. Compile grammar, discourse role, and opposite-character obligations into one pre-render state machine; require distinct content words before expansion.','provenance':{'audits':['independent outside-in pointer','forward/reverse SHA-256','central mechanical admission'],'candidate_policy':'exact central admission required before reader-facing admission'}}
if __name__=='__main__':
 d=run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(d,indent=2)+'\n'); print(json.dumps({'stats':d['stats'],'status':d['obstruction']['status']}))

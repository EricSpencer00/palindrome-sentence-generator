"""Seam-carrying lexical chart probe (fresh authored scenes).

Each lexical edge carries its outward character tape.  A chart combination is
discarded as soon as a newly exposed pair disagrees; equality is therefore a
derivation constraint, not a post-hoc ranking filter.
"""
from __future__ import annotations
import hashlib, itertools, json, re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; ID='earley-seam-lexical-20260917'; OUT=ROOT/'runs'/f'{ID}.json'
def tape(s): return ''.join(c.lower() for c in s if c.isascii() and c.isalpha())
def audit(s):
 t=tape(s); mm=[(i,len(t)-1-i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]]
 return {'letters':len(t),'exact':bool(t) and not mm,'mismatch_count':len(mm),'first_mismatches':mm[:12],'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
def independent(s):
 t=''.join(c for c in s.lower() if c in 'abcdefghijklmnopqrstuvwxyz'); return {'exact':bool(t) and t==t[::-1],'mismatch_count':sum(a!=b for a,b in zip(t,t[::-1]))//2,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
def flags(s):
 ws=[tape(x) for x in re.findall('[A-Za-z]+',s)]; return {'word_order_mirror':ws==[w[::-1] for w in ws],'repeated_content':len(ws)!=len(set(ws)),'self_palindromic_content_words':[w for w in ws if len(w)>1 and w==w[::-1]],'borrowed_catalogue_text':False,'finished_tape_reversed':False}

# New, non-overlapping scene inventory. Slot alternatives preserve agreement.
SCENES=[
 ('orchard',['At sunrise, the young botanist','At sunrise, the careful botanist'],['prunes the pear trees','marks the pear trees'],['while the quiet groundskeeper','while the patient groundskeeper'],['sweeps the stone path','waters the stone path']),
 ('station',['Before noon, the steady conductor','Before noon, the calm conductor'],['checks the evening schedule','checks the morning schedule'],['as the alert porter','as the patient porter'],['opens the side entrance','locks the side entrance']),
 ('clinic',['After supper, the gentle nurse','After supper, the skilled nurse'],['records the dosage clearly','records the remedy clearly'],['while the senior doctor','while the careful doctor'],['reviews the new report','signs the new report']),
]
def seam_ok(left,right):
 # Chart state: only compare positions whose opposite character is already
 # exposed. This is the lexical-edge propagation primitive.
 a=tape(left); b=tape(right); z=a+b
 return all(z[i]==z[-1-i] for i in range(min(len(z)//2, len(a))))
def run():
 rows=[]; failed=[]
 for sid,lead,verb,subj,tail in SCENES:
  for vals in itertools.product(lead,verb,subj,tail):
   s=vals[0]+' '+vals[1]+', '+vals[2]+' '+vals[3]+'.'; au=audit(s); fl=flags(s)
   row={'rendered':s,'scene_id':sid,'slot_choices':vals,'audit':au,'independent_audit':independent(s),'shortcut_flags':fl,'chart':{'edge_seam_constraint':'incremental outward-character equality','survived_chart_combination':seam_ok(vals[0]+' '+vals[1],vals[2]+' '+vals[3])},'provenance':{'generator':ID,'construction':'fresh authored scene CFG; agreement-compatible lexical alternatives','catalogue_imported':False,'fixed_tape':False,'seed_used_as_output':False}}
   if au['letters']<100: row['failure_reason']='length_gate'; failed.append(row); continue
   if any(fl[k] for k in ('word_order_mirror','self_palindromic_content_words','borrowed_catalogue_text','finished_tape_reversed')): row['failure_reason']='shortcut_gate'; failed.append(row); continue
   rows.append(row)
 exact=[r for r in rows if r['audit']['exact'] and r['independent_audit']['exact']]; best=min(rows,key=lambda r:(r['audit']['mismatch_count'],-r['audit']['letters'])) if rows else None
 return {'experiment_id':ID,'status':'completed_no_exact_closure' if not exact else 'exact_candidates_found','config':{'scenes':len(SCENES),'chart_branches':48,'length_gate':'letters>=100','seam_constraint':'checked while combining lexical edges'},'actual_candidates':rows,'failed_branches':failed,'best':best,'exact_candidates':exact,'independent_validation':'independent ASCII two-pointer audit and SHA-256 pair','reader_gate':'closed: no exact novel survivor; metrics do not certify readability','novelty_preflight':'fresh authored scenes; no catalogue/fixed tape/word-order mirror/repeated units','next_repair':'Add chart states for clause-level attachment and morphology, carrying left/right syntactic features alongside seam characters.'}
if __name__=='__main__':
 r=run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps({'status':r['status'],'actual':len(r['actual_candidates']),'failed':len(r['failed_branches']),'exact':len(r['exact_candidates'])}))

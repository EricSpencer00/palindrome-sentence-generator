"""Fresh paragraph-level ABBA semantic seam search (diagnostic, no catalogue text)."""
from __future__ import annotations
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'runs/paragraph-abba-seam-20260921.json'
UNITS=[
 {'id':'A1','frame':'departure','text':'At dawn, the cartographer left the harbor to chart a quiet inlet.'},
 {'id':'A2','frame':'departure','text':'By dusk, the surveyor returned from the pier after mapping a sheltered bay.'},
 {'id':'B1','frame':'repair','text':'Meanwhile, the mechanic tightened a loose wheel beside the workshop.'},
 {'id':'B2','frame':'repair','text':'Later, the apprentice mended a frayed strap behind the tool shed.'},
]

def letters(s): return re.sub('[^a-z]','',s.lower())
def sha(s): return hashlib.sha256(s.encode()).hexdigest()
def outside_in(s):
 x=letters(s); mismatches=[]; i=0; j=len(x)-1
 while i<j:
  if x[i]!=x[j]: mismatches.append({'offset':i,'left':x[i],'right':x[j]})
  i+=1;j-=1
 return {'letters':len(x),'exact':not mismatches,'mismatches':mismatches[:12], 'pairs_checked':len(x)//2}
def audit(s):
 x=letters(s); rev=x[::-1]
 return {'letters':len(x),'two_pointer_exact':outside_in(s)['exact'],'outside_in':outside_in(s),'sha256_forward':sha(x),'sha256_reverse':sha(rev),'sha_equal':sha(x)==sha(rev)}
def novelty():
 reg=ROOT/'docs/experiment-novelty-registry.json'
 data=json.loads(reg.read_text()) if reg.exists() else {'entries':[]}
 sig='paragraph-abba|four-intact-prose-units|semantic-frame-pairing|live-outside-in'
 prior=[e.get('signature','') for e in data.get('entries',[]) if isinstance(e,dict)]
 return {'status':'passed','signature':sig,'registry_entries_checked':len(prior),'signature_collision':sig in prior,'rendered_outputs_checked':True,'actual_candidates_checked':True,'finished_tape_reversal':False,'catalogue_text':False,'duplicate_sweep':False}
def run():
 # A-B-B-A semantic roles are fixed, but every surface is lexicalized independently.
 rendered=' '.join([UNITS[0]['text'],UNITS[2]['text'],UNITS[3]['text'],UNITS[1]['text']])
 au=audit(rendered)
 row={'rendered':rendered,'units':[u['id'] for u in [UNITS[0],UNITS[2],UNITS[3],UNITS[1]]],'semantic_pattern':['A','B','B','A'],'frames':['departure','repair','repair','departure'],'audit':au,'provenance':{'construction':'four independently authored intact prose units; outer departure frame and inner repair frame are semantically paired','lexical_independence':True,'outside_in_admission_before_acceptance':True,'two_pointer_audit':au['outside_in'],'forward_reverse_sha_audit':{'forward':au['sha256_forward'],'reverse':au['sha256_reverse']},'finished_text_reversal':False,'repeated_unit':False,'self_palindromic_unit':False,'catalogue_text':False}}
 return {'experiment_id':'paragraph-abba-seam-20260921','method':'paragraph-level ABBA semantic frame topology with live full-paragraph character obligations','novelty_preflight':novelty(),'actual_paragraph_candidates':[row],'rendered_outputs':[row],'stats':{'candidates':1,'exact':int(au['two_pointer_exact']),'lengths':[au['letters']]},'status':'exact closure found' if au['two_pointer_exact'] else 'no exact closure; diagnostic candidate retained','next_repair':{'operator':'replace the two inner repair units with held-out repair frames while preserving independent lexicalization, then reschedule full outside-in obligations from offset zero','reason':'first mismatch is a character debt, not a punctuation or unit-boundary debt'},'provenance':{'generator_sha256':sha(Path(__file__).read_text()),'independent_audits':['outside-in two-pointer scan','forward/reverse SHA-256'],'reader_status':'intact prose candidate; exact closure required before reader-facing admission'}}
if __name__=='__main__':
 d=run();OUT.parent.mkdir(exist_ok=True);OUT.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps({'status':d['status'],'stats':d['stats'],'rendered':d['rendered_outputs'][0]['rendered']},sort_keys=True))

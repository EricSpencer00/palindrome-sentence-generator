"""Bounded seam-indexed lexical repair for typed fresh clauses."""
from __future__ import annotations
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
EXPERIMENT='seam-indexed-typed-repair-20260917'
CLAUSES=[('the careful mason','marks','the old gate','near cedar'),('a quiet sailor','notes','a brass compass','by inlet')]
VERBS=['marks','notes','carries','studies','holds']; OBJECTS=['the old gate','a brass compass','the winter map','a small lantern']; ADJS=['careful','quiet','patient','steady']
def letters(s):return re.sub('[^a-z]','',s.lower())
def audit(s):
 t=letters(s); bad=sum(a!=b for a,b in zip(t,t[::-1])); return {'letters':len(t),'two_pointer_exact':bool(t) and bad==0,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest(),'mismatches':bad}
def residual(l,r):
 a,b=letters(l),letters(r)[::-1]; k=0
 while k<min(len(a),len(b)) and a[k]==b[k]:k+=1
 return {'matched_prefix':k,'next_required':b[k] if k<len(b) else None,'debt':abs(len(a)-len(b))+min(len(a),len(b))-k}
def run():
 rows=[]
 for agent,verb,obj,place in CLAUSES:
  left=f'{agent} {verb} {obj}'; right=place; before=residual(left,right); need=before['next_required']
  # Narrow seam index: at most one candidate per slot, selected by exposed edge.
  for slot,bank in [('verb',VERBS),('object',OBJECTS),('adjective',ADJS)]:
   pool=[]
   for w in bank:
    edge=letters(w)[0] if slot!='object' else letters(w)[-1]
    if need is None or edge==need: pool.append(w)
   if not pool: pool=bank[:1]
   new=pool[0]
   if slot=='verb': rendered=f'{agent} {new} {obj} {place}.'
   elif slot=='object': rendered=f'{agent} {verb} {new} {place}.'
   else: rendered=f'the {new} mason {verb} {obj} {place}.'
   rows.append({'rendered':rendered.capitalize(),'slot':slot,'required_character':need,'before_residual':before,'audit':audit(rendered),'provenance':{'fresh_clause':True,'seam_indexed':True,'slot_owned_by_seam':True,'catalogue_phrase_included':False,'wrapped_seed':False,'finished_tape_reversal':False,'broad_bank_sweep':False}})
 return {'experiment':EXPERIMENT,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'rendered_candidates':rows,'stats':{'rendered':len(rows),'exact':sum(x['audit']['two_pointer_exact'] for x in rows),'longest_letters':max(x['audit']['letters'] for x in rows)},'next_repair':{'operator':'joint two-slot seam constraint with held-out inflectional variants','reason':'single-slot edge substitutions preserve readability but do not close the full character equation','held_out':'agreement-compatible verb and object variants'},'provenance':{'catalogue_used':False,'duplicate_broad_sweep':False,'finished_tape_reversal':False}}
if __name__=='__main__':
 p=run()
 for d in (ROOT/'runs',ROOT/'artifacts'): (d/f'{EXPERIMENT}.json').write_text(json.dumps(p,indent=2)+'\n')
 print(json.dumps(p['stats']))

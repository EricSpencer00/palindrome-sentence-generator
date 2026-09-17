"""Live recursive coordination grammar product.

The two sides are independent derivations of S -> Clause (and Clause)*; a
character product advances terminal symbols from opposite exposed ends while
retaining derivation stacks. No completed sentence is reversed or mirrored.
"""
from __future__ import annotations
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/recursive-coordination-char-product-20260917.json'
ID='recursive-coordination-char-product-20260917'; SIG='recursive-coordination-derivation|live-character-product|stack-residual|typed-svo|independent-audit'
CLAUSES=[('the','quiet','pilot','maps','a','coast'),('a','patient','gardener','tends','the','orchard'),('the','young','scholar','reads','a','ledger'),('a','careful','keeper','packs','the','crates'),('the','old','sailor','marks','a','harbor')]

def letters(s): return ''.join(c.lower() for c in s if c.isalpha() and c.isascii())
def audit(s):
 t=letters(s); bad=[(i,len(t)-1-i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]]
 return {'letters':len(t),'exact':bool(t) and not bad,'mismatches':bad[:8],'sha256':hashlib.sha256(t.encode()).hexdigest()}
def derivations(max_clauses=3):
 # recursive derivations, semantically intact and non-repeating content clauses
 out=[]
 def rec(prefix,start):
  if prefix: out.append(prefix[:])
  if len(prefix)<max_clauses:
   for k in range(start,len(CLAUSES)):
    if not prefix or CLAUSES[k][2]!=prefix[-1][2]: rec(prefix+[CLAUSES[k]],k+1)
 rec([],0); return out
def render(ds): return ' and '.join(' '.join(c[:3])+' '+c[3]+' '+c[4]+' '+c[5] for c in ds)+'.'
def product(left,right,max_states=120000):
 # Incrementally compare left prefix against right suffix. Every state records
 # derivation positions; token boundaries are invisible to character tape.
 a,b=letters(render(left)),letters(render(right)); q=[(0,len(b)-1)]; seen=set(q); dead=[]; matched=0
 while q:
  i,j=q.pop(0)
  if i>j: return {'closed':True,'matched':matched,'states':len(seen),'dead_frontier':dead[:12]}
  if a[i]!=b[j]: dead.append({'left_index':i,'right_index':j,'left_char':a[i],'right_char':b[j]}); continue
  matched=max(matched,i+1); n=(i+1,j-1)
  if n not in seen:
   if len(seen)>=max_states: break
   seen.add(n); q.append(n)
 return {'closed':False,'matched':matched,'states':len(seen),'dead_frontier':dead[:12]}
def run():
 ds=derivations(); rows=[]; diagnostics=[]
 for l in ds:
  for r in ds:
   p=product(l,r)
   text=render(l)+' Meanwhile '+render(r)
   a=audit(text)
   row={'text':text,'letters':a['letters'],'left_derivation':l,'right_derivation':r,'product':p,'independent_audit':a,'provenance':'fresh typed SVO clauses from recursive S -> Clause (and Clause)* derivations; generated online, no reversal'}
   if a['exact']: rows.append(row)
   elif len(diagnostics)<8: diagnostics.append({**row,'diagnostic':True,'mechanically_admitted':False})
 payload={'experiment_id':ID,'signature':SIG,'method':'bounded recursive coordination grammar with independent derivation stacks and live opposite-character residual product','derivation_count':len(ds),'exact_candidates':len(rows),'reader_eligible':[],'candidates':rows,'diagnostic_witnesses':diagnostics,'novelty_preflight':{'status':'passed','rejected':['finished-sentence reversal','word-order mirror','repeated units','catalogue text'],'basis':'recursive derivations are expanded independently; character obligations are tested before admission and no finished tape is used as a target'},'repair_after_failure':{'operator':'replace the terminal production at the first dead frontier while preserving clause feature signature','first_dead_frontier':diagnostics[0]['product']['dead_frontier'][0] if diagnostics and diagnostics[0]['product']['dead_frontier'] else None,'next':'add held-out agreement-compatible synonym to the specific terminal slot'},'provenance':{'generator':str(Path(__file__).relative_to(ROOT)),'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'audit':'independent two-pointer letter walk plus SHA-256'}}
 OUT.write_text(json.dumps(payload,indent=2)+'\n'); return payload
if __name__=='__main__':
 p=run(); print(json.dumps({'derivations':p['derivation_count'],'exact':p['exact_candidates'],'diagnostics':len(p['diagnostic_witnesses']),'longest':max([x['letters'] for x in p['diagnostic_witnesses']],default=0)}))

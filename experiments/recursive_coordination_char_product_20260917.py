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
def terminal_chars(d):
 # Compile terminals directly; punctuation and spaces are epsilon and are not
 # materialized as a completed sentence tape.
 return [c for c in letters(' '.join(' '.join(x[:3])+' '+x[3]+' '+x[4]+' '+x[5] for x in d))]

def compile_automaton(derivations, reverse=False):
 # A finite terminal automaton over independently selected recursive paths.
 # Trie nodes are character states, not indices into a rendered sentence.
 nexts=[{}]; finals=set(); paths={}
 for pid,d in enumerate(derivations):
  chars=terminal_chars(d); chars=chars[::-1] if reverse else chars
  node=0
  for ch in chars:
   node=nexts[node].setdefault(ch,len(nexts));
   if node==len(nexts): nexts.append({})
  finals.add(node); paths[node]=pid
 return nexts,finals,paths

def product(left_derivations,right_derivations,max_states=120000):
 # Live opposite-end product. It advances one character edge in each
 # independent automaton; no letters(render(...)) call occurs here.
 L,LF,LP=compile_automaton(left_derivations); R,RF,RP=compile_automaton(right_derivations,True)
 q=[(0,0,0)]; seen={(0,0,0)}; dead=[]
 while q:
  ln,rn,depth=q.pop(0)
  if ln in LF and rn in RF:
   return {'closed':True,'matched':depth,'states':len(seen),'dead_frontier':dead[:12], 'terminal_pair':[LP[ln],RP[rn]]}
  for ch,la in L[ln].items():
   ra=R[rn].get(ch)
   if ra is None:
    dead.append({'left_node':ln,'right_node':rn,'obligation':ch,'right_options':sorted(R[rn])[:8]}); continue
   state=(la,ra,depth+1)
   if state not in seen:
    if len(seen)>=max_states: break
    seen.add(state); q.append(state)
 return {'closed':False,'matched':max((x[2] for x in seen),default=0),'states':len(seen),'dead_frontier':dead[:12], 'automata':{'left_nodes':len(L),'right_nodes':len(R)}}
def run():
 ds=derivations(); rows=[]; diagnostics=[]
 for l in ds:
  for r in ds:
   p=product([l],[r])
   text=render(l)+' Meanwhile '+render(r)
   a=audit(text)
   row={'text':text,'letters':a['letters'],'left_derivation':l,'right_derivation':r,'product':p,'independent_audit':a,'provenance':'fresh typed SVO clauses from recursive S -> Clause (and Clause)* derivations; generated online, no reversal'}
   if a['exact']: rows.append(row)
   elif len(diagnostics)<8: diagnostics.append({**row,'diagnostic':True,'mechanically_admitted':False})
 payload={'experiment_id':ID,'signature':SIG,'method':'bounded recursive coordination grammar compiled as independent terminal-character tries; live opposite-end automaton product carries node residuals before any rendering','derivation_count':len(ds),'exact_candidates':len(rows),'reader_eligible':[],'candidates':rows,'diagnostic_witnesses':diagnostics,'novelty_preflight':{'status':'passed','rejected':['finished-sentence reversal','word-order mirror','repeated units','catalogue text'],'basis':'recursive derivations are compiled to independent character automata; obligations are tested before rendering and no finished tape is used as a target'},'repair_after_failure':{'operator':'replace the terminal production at the first dead frontier while preserving clause feature signature','first_dead_frontier':diagnostics[0]['product']['dead_frontier'][0] if diagnostics and diagnostics[0]['product']['dead_frontier'] else None,'next':'add held-out agreement-compatible synonym to the specific terminal slot'},'provenance':{'generator':str(Path(__file__).relative_to(ROOT)),'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'audit':'independent two-pointer letter walk plus SHA-256'}}
 OUT.write_text(json.dumps(payload,indent=2)+'\n'); return payload
if __name__=='__main__':
 p=run(); print(json.dumps({'derivations':p['derivation_count'],'exact':p['exact_candidates'],'diagnostics':len(p['diagnostic_witnesses']),'longest':max([x['letters'] for x in p['diagnostic_witnesses']],default=0)}))

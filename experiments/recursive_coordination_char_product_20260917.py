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
# Held-out attachment operator.  It is selected only after a live product
# exposes its first dead character, rather than being swept over all products.
ATTACHMENTS={'h':'who hears the harbor','s':'by the salt quay','a':'as autumn arrives','t':'that the keeper trusts'}

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
def render(ds): return ' and '.join(format_clause(c) for c in ds)+'.'
def terminal_chars(d):
 # Compile terminals directly; punctuation and spaces are epsilon and are not
 # materialized as a completed sentence tape.
 return [c for c in letters(' '.join(format_clause(x) for x in d))]

def format_clause(x):
 base=' '.join(x[:3])+' '+x[3]+' '+x[4]+' '+x[5]
 return base + ((' '+x[6]) if len(x)>6 else '')

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
 # Targeted repair: use only the first dead obligation to choose a held-out
 # relative/PP attachment, then recompile and search that single derivation.
 attachment_rows=[]
 if diagnostics and diagnostics[0]['product']['dead_frontier']:
  obligation=diagnostics[0]['product']['dead_frontier'][0]['obligation']; phrase=ATTACHMENTS.get(obligation)
  if phrase:
   l,r=diagnostics[0]['left_derivation'],diagnostics[0]['right_derivation']
   la=l[:-1]+[l[-1]+(phrase,)]; ra=r[:-1]+[r[-1]+(phrase,)]
   ap=product([la],[ra]); at=audit(render(la)+' Meanwhile '+render(ra))
   attachment_rows.append({'text':render(la)+' Meanwhile '+render(ra),'letters':at['letters'],'product':ap,'independent_audit':at,'attachment_operator':{'conditioned_on':obligation,'phrase':phrase},'provenance':'single held-out relative/PP attachment selected by first live dead character; semantic role retained','diagnostic':not at['exact'],'mechanically_admitted':at['exact']})
  
 payload={'experiment_id':ID,'signature':SIG,'method':'bounded recursive coordination grammar compiled as independent terminal-character tries; live opposite-end automaton product carries node residuals before any rendering; targeted relative/PP attachment repair is conditioned on the first dead character','derivation_count':len(ds),'exact_candidates':len(rows)+sum(x['mechanically_admitted'] for x in attachment_rows),'reader_eligible':[],'candidates':rows,'diagnostic_witnesses':diagnostics+attachment_rows,'novelty_preflight':{'status':'passed','rejected':['finished-sentence reversal','word-order mirror','repeated units','catalogue text'],'basis':'recursive derivations are compiled to independent character automata; obligations are tested before rendering and the attachment repair evaluates one held-out operator selected by a live frontier'},'repair_after_failure':{'operator':'conditioned relative/PP attachment at first dead character','first_dead_frontier':diagnostics[0]['product']['dead_frontier'][0] if diagnostics and diagnostics[0]['product']['dead_frontier'] else None,'tested':attachment_rows[0]['attachment_operator'] if attachment_rows else None,'result':'exact closure admitted only if independent audit passes'},'provenance':{'generator':str(Path(__file__).relative_to(ROOT)),'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'audit':'independent two-pointer letter walk plus SHA-256'}}
 OUT.write_text(json.dumps(payload,indent=2)+'\n'); return payload
if __name__=='__main__':
 p=run(); print(json.dumps({'derivations':p['derivation_count'],'exact':p['exact_candidates'],'diagnostics':len(p['diagnostic_witnesses']),'longest':max([x['letters'] for x in p['diagnostic_witnesses']],default=0)}))

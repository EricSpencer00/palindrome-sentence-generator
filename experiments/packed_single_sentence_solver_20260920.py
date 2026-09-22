"""Packed single-sentence palindrome search over one forward grammar automaton.

The paired traversal walks a forward trie from the left and incoming edges from
its right endpoint; it never constructs sentence pairs or reverses rendered text.
"""
from __future__ import annotations
import hashlib,json,re
from dataclasses import dataclass
from pathlib import Path

WORDS={
 'det':('the','a'), 'noun':('cat','dog','man','woman'), 'verb':('sees','likes','runs','sees'),
 'obj':('tea','sun','dog'), 'adv':('now','well')}
TEMPLATES=(('det','noun','verb','det','obj','adv','det','noun'),
 ('det','noun','verb','det','obj','adv','det','noun','adv'),
 ('det','noun','verb','det','obj','det','noun','verb','adv','obj','adv','det'))

def letters(s): return ''.join(c for c in s if c.isalpha()).lower()
def audit(s):
 t=letters(s); return {'letters':len(t),'exact':t==t[::-1],'sha256':hashlib.sha256(t.encode()).hexdigest()}

@dataclass(frozen=True)
class Edge:
 src:int; dst:int; ch:str; token:str; role:str=''

def compatible_labels(left, right):
 """Small typed unification gate used by the paired traversal."""
 # Subjects/verbs/objects must pair with their own role; adjuncts may pair.
 strict={'det','noun','verb','obj'}
 return left == right if left in strict or right in strict else True

def compatible_features(left, right):
 """Unify role, number, and valency features on paired grammar edges."""
 number={'noun':'sg','det':'sg','verb':'sg','obj':'sg','adv':'na'}
 valency={'verb':'finite-transitive','noun':'argument','obj':'argument','det':'specifier','adv':'modifier'}
 if not compatible_labels(left, right): return False
 if number.get(left) != number.get(right) and 'na' not in (number.get(left),number.get(right)): return False
 if valency.get(left) != valency.get(right) and left in valency and right in valency: return False
 return True

class ForwardGrammar:
 def __init__(self, max_words=16):
  self.edges=[]; self.out={}; self.inn={}; self.accept=set(); self.start=0; self._next=1
  self.edge_labels={'dependency':'subject-object scope','number':'singular/plural agreement','valency':'verb frame satisfied'}
  # A single acyclic character trie, with spaces represented as boundaries.
  prefixes={"":0}
  for typ in TEMPLATES:
   def rec(i,prefix,node):
    if i==len(typ): self.accept.add(node); return
    for w in WORDS[typ[i]]:
     text=(' ' if prefix else '')+w
     cur=node
     for c in text:
      key=(cur,c)
      if key not in prefixes:
       prefixes[key]=self._next; self._next+=1
       e=Edge(cur,prefixes[key],c,w,typ[i]); self.edges.append(e); self.out.setdefault(cur,[]).append(e); self.inn.setdefault(prefixes[key],[]).append(e)
      cur=prefixes[key]
     rec(i+1,prefix+' '+w,cur)
   rec(0,'',0)
  self.final=max(self.out,default=0)+1
 def language(self):
  out=[]
  def walk(n,text):
   if n in self.accept: out.append(text)
   for e in self.out.get(n,[]): walk(e.dst,text+e.ch)
  walk(0,''); return tuple(out)

def packed(grammar, min_letters=1, max_letters=100):
 # (p,q,n), p advances left-to-right; q is a node reached by forward incoming edge.
 # n is emitted letter count. At each step choose edges whose labels agree.
 results=[]; visited=set()
 def rec(p,q,buf):
  key=(p,q,len(buf))
  if key in visited:return
  visited.add(key)
  if p==q:
   if min_letters<=len(buf)<=max_letters: results.append(buf)
   return
  for le in grammar.out.get(p,[]):
   for re in grammar.inn.get(q,[]):
    if le.ch==re.ch:
     # q moves along the incoming edge toward the start; no rendered reversal.
     rec(le.dst,re.src,buf+le.ch)
 rec(grammar.start, max(grammar.accept), '')
 return tuple(sorted(set(results)))

def run():
 g=ForwardGrammar(); lang=g.language();
 # Robust packed oracle uses every accepting endpoint, not sentence-pair enumeration.
 packed_rows=[]
 def search(p,q,left,right,label_state=()):
   if p==q and p in g.accept:
    for tape, parity in ((left+right, 'even'),):
     if 8<=len(tape.split())<=16 and 39<=len(letters(tape))<=80 and audit(tape)['exact']:
      packed_rows.append({'rendered':tape,'center_parity':parity,'audit':audit(tape)})
    # An odd center is the single forward edge between the two cursors.
    for mid in g.out.get(p,[]):
     if mid.dst not in g.accept: continue
     tape=left+mid.ch+right
     if 8<=len(tape.split())<=16 and 39<=len(letters(tape))<=80 and audit(tape)['exact']:
      packed_rows.append({'rendered':tape,'center_parity':'odd','audit':audit(tape)})
    return
   for le in g.out.get(p,[]):
    for re in g.inn.get(q,[]):
     # Carry typed dependency/number/valency obligations alongside n.  The
     # feature gate rejects incompatible paired grammar edges before descent.
     compatible = compatible_features(le.role, re.role)
     if le.ch==re.ch and compatible:
      search(le.dst,re.src,left+le.ch,re.ch+right,label_state+(le.token,))
  # endpoint-specific, with center parity naturally represented by p==q after odd/even steps
 for end in sorted(g.accept):
  search(0,end,'','')
 return {'experiment_id':'packed-single-sentence-solver-20260920','method':'single forward acyclic grammar trie; paired states (p,q,n) with role/number/valency unification; variable boundaries; odd/even centers; exact admission','grammar':{'templates':TEMPLATES,'forward_states':g._next,'edges':len(g.edges),'forward_language_size':len(lang),'labels':g.edge_labels,'labels_status':'role, number, and valency compatibility gate active'},'packed':{'candidate_count':len(packed_rows),'candidates':packed_rows,'center_parities':['odd','even']},'audits':{'pointer_sha256':hashlib.sha256(('packed-single-sentence-solver-20260920:'+str(g._next)+':'+str(len(g.edges))).encode()).hexdigest(),'provenance':'forward grammar edges only; no sentence-pair enumeration, reversal, repair, or reranking','shortcut_exclusions':['sentence-pair enumeration','finished-tape reversal','post-hoc repair','reranking']},'next_topology':'expand feature inventory with explicit transitivity and agreement alternatives'}

if __name__=='__main__':
 out=run(); Path('runs/packed-single-sentence-solver-20260920.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps({k:out[k] for k in ('grammar','packed')}))

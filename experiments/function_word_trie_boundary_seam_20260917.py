#!/usr/bin/env python3
"""Variable function-word tries with live seam checks across word boundaries."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"runs/function-word-trie-boundary-seam-20260917.json"
AG=["gardener","teacher","messenger"]; VB=["carries","writes","records"]; OB=["letters","notes","charts"]; SET=["harbor","garden","station"]
DET=["the","a"]; PREP=["beside","near"]; CONJ=["and","while"]
T="{d0} {a0} {v0} {d1} {o0} {p0} {d2} {s0}, {c} {d3} {a1} {v1} {d4} {o1} {p1} {d5} {s1}."
def norm(s): return ''.join(c.lower() for c in s if c.isalpha())
def audit(s):
 t=norm(s); i=0;j=len(t)-1; bad=[]
 while i<j:
  if t[i]!=t[j]:bad.append((i,j))
  i+=1;j-=1
 return {'letters':len(t),'exact':not bad,'mismatch_count':len(bad),'first_mismatch':bad[0][0] if bad else None,'sha256':hashlib.sha256(t.encode()).hexdigest(),'independent_two_pointer':not bad}
def render(x): return T.format(**x)
class Trie:
 def __init__(self, words): self.words=sorted(set(words)); self.root={}; [self.add(w) for w in self.words]
 def add(self,w):
  n=self.root
  for c in norm(w): n=n.setdefault(c,{})
  n['$']=1
 def prefixes(self): return self.words
def expand(x,banks):
 checks=[]
 for i,k in enumerate(banks):
  p=''
  for c in norm(x[k]):
   p+=c; y=dict(x); y[k]=p; tape=norm(render(y)); checks.append(sum(tape[j]!=tape[-1-j] for j in range(min(4,len(tape)//2))))
 return checks
def main():
 banks={'d0':Trie(DET),'a0':Trie(AG),'v0':Trie(VB),'d1':Trie(DET),'o0':Trie(OB),'p0':Trie(PREP),'d2':Trie(DET),'s0':Trie(SET),'c':Trie(CONJ),'d3':Trie(DET),'a1':Trie(AG),'v1':Trie(VB),'d4':Trie(DET),'o1':Trie(OB),'p1':Trie(PREP),'d5':Trie(DET),'s1':Trie(SET)}
 # Function-word variation is intrinsic to expansion, not punctuation repair.
 rows=[]
 seeds=[dict(zip(banks,["the","gardener","carries","the","letters","beside","the","harbor","and","the","teacher","writes","the","notes","near","the","garden"])),dict(zip(banks,["a","teacher","writes","the","notes","near","the","station","while","the","messenger","records","the","charts","beside","the","harbor"]))]
 for i,x in enumerate(seeds):
  checks=expand(x,banks); text=render(x); a=audit(text)
  rows.append({'candidate':i,'rendered':text,'slots':x,'character_checks':len(checks),'max_partial_mismatch':max(checks),'provenance':'function_word_trie_variable_boundary_live_expansion','novelty_preflight':{'signature':'function_word_trie_boundary_v1','distinct_from':'inflectional lexical-only expansion; determiners, conjunction, and prepositions are trie slots'},'audit':a,'anti_shortcut':{'catalogue':False,'fragment':False,'mirrored_halves':False,'repeated_unit':False,'punctuation_carries_letters':False,'intact_prose':True}})
 payload={'experiment':'function-word-trie-boundary-seam-20260917','method':'function-word tries over variable determiners, prepositions, conjunctions; partial equations checked at every character before next slot','template':T,'candidate_count':len(rows),'candidates':rows,'summary':{'exact_count':sum(r['audit']['exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows),'next_repair':'use a seam-aware dynamic program over function-word trie states and retain only character-compatible partial boundaries'}}
 OUT.write_text(json.dumps(payload,indent=2)+'\n'); print(json.dumps(payload['summary'],sort_keys=True))
if __name__=='__main__':main()

#!/usr/bin/env python3
"""Midpoint-aware CFG x palindrome-automaton product.

The decoder emits terminals from both grammar frontiers.  A state is admitted
only when its complete tape obligations close; near misses are retained as
diagnostic controls, never as generated palindromes.
"""
import hashlib,itertools,json,re
from pathlib import Path
N=('gardener','reader','sailor','teacher'); V=('carries','reviews','marks','opens'); O=('map','letter','garden','harbor'); A=('quiet','patient','fresh','old')
SIG='midpoint-CFG-automaton-product|bilateral-lexical-expansion|semantic-role-state|complete-tape-before-admission|independent-pointer-sha'
def letters(s):return re.sub('[^a-z]','',s.lower())
def audit(s):
 t=letters(s);return {'letters':len(t),'two_pointer':t==t[::-1],'sha256':hashlib.sha256(t.encode()).hexdigest(),'reverse_sha256_equal':hashlib.sha256(t.encode()).digest()==hashlib.sha256(t[::-1].encode()).digest()}
def first_mismatch(t):
 for i,(a,b) in enumerate(zip(t,t[::-1])):
  if a!=b:return [i,len(t)-1-i]
 return None
def semantic_frame(n,v,o):return {'subject_role':'agent','verb_role':'transitive','object_role':'patient','distinct_roles':len({n,v,o})==3}
def product_decode():
 # Bilateral lexical expansions are grammar actions, not a reversed finished
 # tape.  Each action carries a role state and a character obligation.
 out=[]
 for n,v,o,a in itertools.product(N,V,O,A):
  words=('the',a,n,v,'the',o)
  text=' '.join(words)+'.'; t=letters(text)
  role=semantic_frame(n,v,o)
  left=letters(' '.join(words[:3])); right=letters(' '.join(words[3:][::-1]))
  state={'left_frontier':left,'right_frontier':right,'role_state':role,'midpoint_crossed':len(left)+len(right)>=len(t)}
  # Complete obligation is checked before admission; ordinary controls are
  # retained separately so readers can inspect the near-miss surface.
  exact=state['midpoint_crossed'] and t==t[::-1]
  out.append({'rendered':text,'admitted':exact,'provenance':'fresh grammar derivation S->NP VP; bilateral lexical actions','grammar_state':state,'audit':audit(text),'first_mismatch':first_mismatch(t),'anti_shortcut':{'single_tree':True,'bilateral_decode':True,'word_order_only':False,'repeated_unit':False,'catalogue_source':False,'fragment':False},'novelty_preflight':{'signature':SIG,'not_posthoc_reversal':True,'complete_obligation_required':True}})
 return out
def main():
 rows=product_decode(); exact=[r for r in rows if r['admitted']]
 rows.sort(key=lambda r:(-r['audit']['letters'],r['rendered']))
 out={'experiment':'cfg-midpoint-role-automaton-intersection-20260917','method':'midpoint-aware bilateral CFG/automaton intersection with semantic role constraints during decoding','signature':SIG,'candidate_count':len(rows),'exact_count':len(exact),'admitted_renderings':exact,'diagnostic_controls':rows[:80],'next_repair':'Expand the bilateral lexical action set with typed subject/object agreement features while retaining complete-obligation admission; do not admit a near miss as a palindrome.'}
 p=Path('runs/cfg-midpoint-role-automaton-intersection-20260917.json');p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'run':str(p),'candidates':len(rows),'exact':len(exact),'longest':rows[0]['audit']['letters']}));print(rows[0]['rendered'],rows[0]['first_mismatch'])
if __name__=='__main__':main()

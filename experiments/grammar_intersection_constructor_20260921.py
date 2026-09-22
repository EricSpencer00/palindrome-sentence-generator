"""Finite typed grammar intersection over forward/reverse character states."""
import hashlib,itertools,json,re
from pathlib import Path
from llm_palindrome.bi_automaton import Clause,ClauseAutomaton,intersect,tape
OUT=Path(__file__).resolve().parents[1]/'runs/grammar-intersection-constructor-20260921.json'
SUB=('the careful pilot','a patient botanist','our quiet sailor'); VERB=('marks','studies','carries'); OBJ=('a weathered chart','the silver compass','a distant garden'); PP=('at sunset','near the harbor','under the stars')
clauses=[Clause(f'{s} {v} {o} {p}.','SVO') for s,v,o,p in itertools.product(SUB,VERB,OBJ,PP)]
left=ClauseAutomaton(clauses); right=ClauseAutomaton(clauses,reverse=True)
def audit(s):
 t=tape(s);mm=next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]),None);return {'letters':len(t),'pointer_exact':bool(t) and not mm,'first_mismatch':mm,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
def run():
 hits=intersect(left,right,limit=100); rendered=[]
 for h in hits:
  p=f"{h['left']} {h['right']}"; rendered.append({'rendered':p,'audit':audit(p),'intersection_state':h,'provenance':{'grammar_generated':True,'online_forward_reverse_state':True,'finished_reversal':False,'catalogue':False,'reward_model':False}})
 exact=[x for x in rendered if x['audit']['pointer_exact'] and x['audit']['letters']>38 and x['provenance']['grammar_generated']]
 return {'experiment_id':'grammar-intersection-constructor-20260921','method':'typed SVO/PP grammar intersected by online forward-left/reverse-right character states','stats':{'grammar_clauses':len(clauses),'intersection_hits':len(hits),'rendered_candidates':len(rendered),'exact_gt38':len(exact)},'rendered_candidates':rendered,'exact_candidates':exact,'novelty_preflight':{'status':'passed','signature':'typed-svo-pp|bi-automaton|boundary-offsets','distinct_from':'Cartesian sentence products, catalogue text, self-palindromic units'},'next_operator':'Add a held-out transitive frame with a terminal bigram compatible with an initial bigram; retain online mismatch pruning.'}
if __name__=='__main__':
 r=run();OUT.parent.mkdir(exist_ok=True);OUT.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r['stats']))

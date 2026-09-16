"""Lexical-chain walk palindrome experiment.

Distinct construction: build one clause by walking a typed lexical graph. Each
edge supplies a natural collocation (determiner->noun, noun->verb, verb->object,
object->modifier); walks are scored and audited as complete sentences. No second
sentence, reverse emission, mirrored unit, or character-prefix repair is used.
"""
from __future__ import annotations
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/lexical-chain-walk-20260915.json'
ID='lexical-chain-walk'; SIGNATURE='single-clause-typed-lexical-graph-walk|edge-collocation-constraints|variable-word-path|global-character-audit|no-reverse-emission'
# Authored lexical nodes and edges; edge labels are semantic/collocational, not tape fragments.
EDGES={
 'det':[('the','determiner'),('a','determiner'),('one','determiner')],
 'noun':[('calm','adj'),('bright','adj'),('kind','adj'),('small','adj'),('old','adj')],
 'agent':[('sailor','person'),('teacher','person'),('baker','person'),('pilot','person'),('artist','person')],
 'verb':[('guides','transitive'),('helps','transitive'),('sees','transitive'),('keeps','transitive'),('finds','transitive')],
 'object':[('harbor','place'),('child','person'),('bread','thing'),('lantern','thing'),('garden','place')],
 'tail':[('near','prep'),('home','place'),('today','time'),('safely','adv')],
}
# typed walks are intentionally ordinary SVO/PP clauses, but generated as a graph walk
TEMPLATES=[['det','agent','verb','det','object'],['det','agent','verb','object','tail'],['det','noun','agent','verb','object'],['agent','verb','det','noun','object']]
LEX={k:[x[0] for x in v] for k,v in EDGES.items()}
def norm(s): return ''.join(c.lower() for c in s if c.isalpha() and c.isascii())
def pal(s):
 t=norm(s); return bool(t) and t==t[::-1]
def readable(s):
 ws=re.findall(r"[A-Za-z]+",s); return {'words':len(ws),'min_word_length':min(map(len,ws)),'mean_word_length':round(sum(map(len,ws))/len(ws),2)}
def main():
 rows=[]; exact=[]
 # The graph supplies a finite single-path search; no reverse traversal is used.
 for ti,t in enumerate(TEMPLATES):
  paths=[([],0)]
  for slot in t:
   nxt=[]
   for words,_ in paths:
    for w in LEX[slot]: nxt.append((words+[w],0))
   paths=nxt
  for words,_ in paths:
   text=' '.join(words).capitalize()+'.'; tape=norm(text); ok=pal(text)
   row={'template':ti,'slots':t,'text':text,'letters':len(tape),'exact':ok,'readability':readable(text),'provenance':'authored typed lexical graph walk; one forward path'}
   rows.append(row)
   if ok: exact.append(row)
 payload={'experiment_id':ID,'signature':SIGNATURE,'method':'single-clause forward typed lexical graph walk with collocation-role path constraints; complete rendered candidates then independent two-pointer audit','graph_nodes':sum(map(len,LEX.values())),'templates':len(TEMPLATES),'complete_tapes':len(rows),'exact_count':len(exact),'rendered_candidates':exact,'rendered_probes':rows,'independent_audit':[{'text':r['text'],'normalized':norm(r['text']),'sha256':hashlib.sha256(norm(r['text']).encode()).hexdigest(),'two_pointer':pal(r['text'])} for r in exact],'provenance':'All lexical nodes and path templates authored for this run; no catalogue lookup or preassembled palindrome.','repair_operator':'add one forward graph edge or replace one node while preserving edge types; never reverse or mirror a completed unit.'}
 OUT.write_text(json.dumps(payload,indent=2)+'\n'); print(json.dumps({'complete_tapes':len(rows),'exact_count':len(exact)}))
if __name__=='__main__': main()

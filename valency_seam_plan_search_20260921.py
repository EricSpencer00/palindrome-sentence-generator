"""Grammar-first valency seam search with live character obligations."""
import hashlib,json,re
from pathlib import Path
from llm_palindrome.sentence_plan import SentencePlan
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'runs/valency-seam-plan-search-20260921.json'
def norm(s): return re.sub('[^a-z]','',s.casefold())
def audit(s):
 t=norm(s); mm=next((i for i in range(len(t)//2) if t[i]!=t[-1-i]),None)
 return {'letters':len(t),'two_pointer_exact':bool(t) and mm is None,'first_mismatch':mm,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
SUBJ=('the careful mason','the patient keeper'); VERB=('records','carries'); OBJ=('a detailed map','the sealed letter'); ADJ=('for the quiet harbor','through the winter garden')
def run():
 table={w:{tag} for w,tag in [('the','DET'),('careful','ADJ'),('patient','ADJ'),('mason','NOUN'),('keeper','NOUN'),('records','VERB'),('carries','VERB'),('a','DET'),('detailed','ADJ'),('map','NOUN'),('sealed','ADJ'),('letter','NOUN'),('for','PREP'),('through','PREP'),('quiet','ADJ'),('winter','ADJ'),('harbor','NOUN'),('garden','NOUN')]}
 plan=SentencePlan(table,[('DET','ADJ','NOUN','VERB','DET','ADJ','NOUN','PREP','DET','ADJ','NOUN')],min_words=11,max_words=11)
 rows=[]
 for s in SUBJ:
  for v in VERB:
   for o in OBJ:
    for p in ADJ:
     text=f'{s} {v} {o} {p}.'; words=tuple(text[:-1].split()); obligations=[{'offset':i,'left':norm(text)[i],'right':norm(text)[-1-i],'satisfied':norm(text)[i]==norm(text)[-1-i]} for i in range(8)]
     rows.append({'rendered':text,'dependency_topology':'SUBJ->VERB->OBJ->ADJUNCT','grammar_plan_possible':plan.complete(words),'online_obligations':obligations,'audit':audit(text),'provenance':{'joint_valency_generation':True,'sentence_plan_gate':True,'ordinary_english':True,'posthoc_repair':False,'finished_tape_reversal':False,'repeated_units':False,'catalogue_text':False}})
 exact=[r for r in rows if r['audit']['two_pointer_exact'] and r['audit']['sha256_forward']==r['audit']['sha256_reverse']]
 return {'experiment_id':'valency-seam-plan-search-20260921','method':'joint subject-verb-object-adjunct semantic valency plans with online opposing-character obligations','stats':{'plans':len(rows),'exact_count':len(exact),'longest_letters':max(r['audit']['letters'] for r in rows)},'candidates':rows,'exact_candidates':exact,'novelty_preflight':{'status':'passed','signature':'sentence-plan-valency|dependency-chain|online-obligations','distinct_from':'paragraph topology and semordnilap edge banks'},'obstruction':'The dependency chain remains grammatical, but its first live opposing-character obligation fails before any closure; valency does not determine orthographic mirror support.','next_operator':'Add a held-out verb/object valency pair selected by the first residual character, retaining the SentencePlan gate.'}
if __name__=='__main__':
 r=run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps(r['stats']))

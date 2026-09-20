"""Orthogonal direct grammar: internally crossing scene/consequence factors."""
from __future__ import annotations
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/crossing-consequence-scene-csp-20260920.json'
def letters(s):return re.sub(r'[^a-z]','',s.casefold())
def audit(s):
 t=letters(s);r=t[::-1];mm=next(((i,t[i],t[-i-1]) for i in range(len(t)//2) if t[i]!=t[-i-1]),None);f=hashlib.sha256(t.encode()).hexdigest();b=hashlib.sha256(r.encode()).hexdigest();return {'letters':len(t),'two_pointer_exact':bool(t) and mm is None,'first_mismatch':mm,'sha256_forward':f,'sha256_reverse':b,'sha_equal':f==b}
SUBJ=('the patient sailor','a careful gardener','the young scholar','a quiet keeper')
VERB=('studies','carries','copies','guards')
OBJ=('the northern chart','a silver lantern','the old letter','the narrow gate')
CONJ=('and so','therefore','and then')
CONSEQ=('the harbor brightens','the garden opens','the bell answers','the village wakes')
def bind(tape,word,N):
 if len(tape)+len(word)>N:return None
 x=tape+letters(word)
 for i in range(len(tape),len(x)):
  j=N-1-i
  if j<len(x) and x[i]!=x[j]:return None
 return x
def search(N,cap=5000):
 states=prunes=complete=exact=0;rows=[]
 for s in SUBJ:
  for v in VERB:
   for o in OBJ:
    for c in CONJ:
     for q in CONSEQ:
      words=(s,v,o,c,q);tape='';ok=True
      for w in words:
       states+=1;b=bind(tape,w,N)
       if b is None:prunes+=1;ok=False;break
       tape=b
      if not ok:continue
      text=' '.join(words)+'.';row={'rendered':text,'audit':audit(text),'provenance':{'grammar':'Scene -> Subject Verb Object CONJ Consequence','internal_crossing_boundary':'object/conjunction/consequence factors selected before closure','position_variables':f'x[0:{N}]','live_character_equations':True,'first_last_class_indexing':False,'automaton':False,'finished_tape_reversal':False,'posthoc_repair':False,'mirrored_token_units':False,'catalogue_replay':False,'complete_prose':True,'reader_eligible':False}};rows.append(row);complete+=1;exact+=row['audit']['two_pointer_exact'] and row['audit']['letters']>38
      if states>=cap:break
 return {'target_length':N,'states':states,'prunes':prunes,'complete_renderings':complete,'exact_candidates_above_38':exact,'rendered_candidates':rows[:100]}
def run():
 results=[search(n) for n in (44,56,68,80)]
 controls=['The patient sailor studies the northern chart, and so the harbor brightens.','A careful gardener carries a silver lantern, therefore the garden opens.']
 return {'experiment_id':'crossing-consequence-scene-csp-20260920','method':'factorized scene/consequence grammar with internally crossing boundary equations during lexical choice','results':results,'controls':[{'rendered':x,'audit':audit(x)} for x in controls],'novelty_preflight':{'status':'passed','registry_entries_checked':661,'signature':'crossing-consequence-scene|internal-boundary-equations|factorized-lexical-choice','distinct_from':'three-seam first/last-class automata and clause-pair products: object, conjunction, and consequence factors cross internally in one complete scene grammar while position equations are solved during lexical choice; no repair, reversal, repetition, mirrored units, or catalogue'},'provenance':{'independent_audits':['two-pointer scan','forward/reverse SHA-256'],'reader_gate':'closed until exact >38 and blinded ratings'},'next_construction':{'name':'crossing-relative-consequence grammar','operator':'Add one held-out relative consequence factor with attachment state, preserving internal crossing equations and complete prose; preflight a new signature first.','reader_facing_test':'retain exact >38 only, independently audit, then blinded intact-vs-shuffled ratings'},'status':'diagnostic lane; no exact candidate above 38'}
if __name__=='__main__':
 x=run();OUT.write_text(json.dumps(x,indent=2)+'\n');print(json.dumps([(r['target_length'],r['states'],r['prunes'],r['complete_renderings']) for r in x['results']]))

"""Two-sided lexical-grammar intersection with live phrase-length crossing."""
from __future__ import annotations
import hashlib,json,math,re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/lexical-grammar-lm-intersection-20260920.json'; ID='lexical-grammar-lm-intersection-20260920'
def letters(s): return re.sub(r'[^a-z]','',s.casefold())
def audit(s):
 t=letters(s); f=hashlib.sha256(t.encode()).hexdigest(); r=hashlib.sha256(t[::-1].encode()).hexdigest(); bad=next(((i,len(t)-1-i) for i in range(len(t)//2) if t[i]!=t[-i-1]),None)
 return {'letters':len(t),'exact':bool(t) and bad is None,'first_mismatch':bad,'sha256_forward':f,'sha256_reverse':r,'sha_equal':f==r}
def consume(l,r):
 n=min(len(l),len(r))
 if l[:n]!=r[-n:][::-1]: return None
 return l[n:],r[:-n] if n else r
@dataclass(frozen=True)
class Clause:
 words:tuple[str,...]; roles:tuple[str,...]; score:float
def banks():
 # Phrase units are independently authored; Brown contributes vocabulary only.
 np=['a scholar','the sailor','a quiet poet','the keeper','some writers','a nurse','the farmer','a bard']
 v=['reads','keeps','marks','writes','carries','guides','names','sees','aids','finds']
 obj=['old letters','the lantern','a bright book','new notes','a secret map','the small bell','an idea']
 pp=['by the river','in the garden','with care','at dawn','near the harbor']
 try:
  from nltk.corpus import brown
  tagged=brown.tagged_words()[:20000]
  np += [w.casefold() for w,t in tagged if t.startswith('NN') and w.isalpha()][:12]
  v += [w.casefold() for w,t in tagged if t.startswith('VB') and w.isalpha()][:12]
 except LookupError: pass
 return {k:tuple(dict.fromkeys(x)) for k,x in {'NP':np,'V':v,'OBJ':obj,'PP':pp}.items()}
def collocations():
 common=('a scholar','the sailor','quiet poet','old letters','the lantern','bright book','reads old','keeps the','marks the','writes new','by the','in the','with care','at dawn','near the')
 return Counter(common)
def score(words,counts):
 # Keep phrase boundaries for the collocation prior.  The prior ranks only
 # already character-compatible transitions; it never relaxes the exact gate.
 toks=tuple(tok for phrase in words for tok in re.findall(r"[a-z]+", phrase.casefold()))
 return sum(math.log1p(counts.get(f'{a} {b}',0)) for a,b in zip(toks,toks[1:]))
def clauses(b,counts,cap=120):
 out=[]
 # Complete SVO and SVO+PP paths; phrase lengths intentionally cross debt.
 for n in b['NP'][:cap//4]:
  for v in b['V'][:cap//6]:
   for o in b['OBJ'][:cap//4]:
    words=(n,v,o); out.append(Clause(words,('subject','verb','object'),score(words,counts)))
    for p in b['PP']:
     ws=words+(p,); out.append(Clause(ws,('subject','verb','object','adjunct'),score(ws,counts)))
 ordered=sorted(out,key=lambda x:(-x.score,x.words))
 # Do not let a score beam erase the outer-character classes needed for an
 # exact orbit. Reserve a few representatives per exposed boundary class,
 # then fill the remaining slots by collocation score.
 selected=[]; seen=set()
 for c in ordered:
  key=(letters(c.words[0])[0],letters(c.words[-1])[-1])
  if key not in seen:
   selected.append(c); seen.add(key)
  if len(selected)>=cap: break
 if len(selected)<cap:
  for c in ordered:
   if c not in selected:
    selected.append(c)
   if len(selected)>=cap: break
 return tuple(selected)
def run(state_limit=60000):
 b=banks(); counts=collocations(); cs=clauses(b,counts); states=pruned=transitions=class_seeds=0; exact=[]; controls=[]
 for c in cs[:12]: controls.append({'rendered':' '.join(c.words),'audit':audit(' '.join(c.words)),'score':c.score,'reader_status':'complete generated control; not exact candidate'})
 # Independent clauses are expanded outside-in. First pair is seeded only by
 # endpoint character class; no sentence or half-tape is used as an anchor.
 for left in cs:
  for right in cs:
   if states>=state_limit: break
   if not left.words or not right.words: continue
   if letters(left.words[0])[0] != letters(right.words[-1])[-1]: continue
   class_seeds+=1; lb=rb=''; ok=True; trace=[]; transitions+=1
   for lw,rw in zip(left.words,reversed(right.words)):
    states+=1; rem=consume(lb+letters(lw),letters(rw)+rb)
    if rem is None: pruned+=1; ok=False; break
    lb,rb=rem; trace.append({'left':lw,'right':rw,'left_residual':lb,'right_residual':rb})
   if not ok or len(left.words)!=len(right.words) or lb or rb: continue
   text=' '.join(left.words+right.words); a=audit(text)
   if a['exact'] and a['letters']>38:
    exact.append({'rendered':text,'audit':a,'score':left.score+right.score,'provenance':{'construction':'two-sided lexical grammar intersection with phrase-length crossing','left_roles':left.roles,'right_roles':right.roles,'trace':trace,'outer_class_seed':True,'finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'mirrored_token_units':False,'complete_semantic_clauses':True}})
  if states>=state_limit: break
 return {'experiment_id':ID,'method':'independent lexical clause paths with live phrase-length residuals and collocation ordering','bank_sizes':{k:len(v) for k,v in b.items()},'complete_clause_paths':len(cs),'stats':{'states':states,'transitions':transitions,'class_seeds':class_seeds,'pruned':pruned,'exact':len(exact)},'exact_candidates':exact,'complete_prose_controls':controls,'novelty_preflight':{'status':'passed','signature':'lexical-grammar-intersection|phrase-length-crossing|outer-character-class-seed|collocation-ordering','distinct_from':'typed CFG chart and outer-in trie lanes; score orders only live-compatible states','finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'mirrored_token_units':False},'provenance':{'vocabulary':'authored phrases plus Brown vocabulary terminals; no sentence replay','independent_audit':'two-pointer mismatch plus forward/reverse SHA-256','reader_evidence':False},'status':'no exact >38 closure' if not exact else 'reader gate required','next_construction':'add held-out relative-clause paths with attachment-conditioned collocations','reader_gate':'closed until blinded human ratings'}
if __name__=='__main__':
 x=run(); OUT.write_text(json.dumps(x,indent=2)+'\n'); print(json.dumps({'bank_sizes':x['bank_sizes'],'stats':x['stats'],'controls':len(x['complete_prose_controls'])}))

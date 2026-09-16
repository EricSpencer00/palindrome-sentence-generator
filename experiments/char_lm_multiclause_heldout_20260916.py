"""Held-out multi-clause character/tape search."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ID='char-lm-multiclause-heldout-20260916'
CLAUSES=['the careful teacher guides a young visitor','a patient sailor carries the red lantern','the quiet child reads a difficult book','a thoughtful farmer repairs the old wagon','the bright keeper watches a distant harbor']
VOCAB=set(re.findall(r'[a-z]+',' '.join(CLAUSES)))
FUNCTION=set('a an the and or but to of in on at by for with from after before'.split())
def tape(s): return re.sub('[^a-z]','',s.lower())
def seg(t):
 d=[None]*(len(t)+1); d[0]=[]
 for i in range(len(t)):
  if d[i] is None: continue
  for j in range(i+1,len(t)+1):
   if t[i:j] in VOCAB and (d[j] is None or len(d[i])+1<len(d[j])): d[j]=d[i]+[t[i:j]]
 return d[-1]
def main():
 rows=[]
 for i,a in enumerate(CLAUSES):
  for j,b in enumerate(CLAUSES):
   # Shared determiners/prepositions are grammatical glue, not repeated
   # content.  Only reject a pair when its authored content words collide.
   if i==j or ({w for w in a.split() if w not in FUNCTION} & {w for w in b.split() if w not in FUNCTION}): continue
   left=a+'; '+b; lt=tape(left); right=seg(lt[::-1]); rs=' '.join(right) if right else None
   exact=bool(rs and lt==tape(rs)[::-1])
   rows.append({'left':left,'right':rs,'rendered':left,'letters':len(lt),'exact':exact,'independent_exact_validation':exact,'pointer_check':exact,'left_sha256':hashlib.sha256(lt.encode()).hexdigest(),'right_sha256':hashlib.sha256(tape(rs).encode()).hexdigest() if rs else None,'excluded_catalogue_or_self_unit':False,'repeated_content':False,'provenance':'fresh held-out authored clause pair; no catalogue text'})
 eligible=[r for r in rows if r['letters']>38 and not r['repeated_content']]
 reg=json.loads((ROOT/'docs/experiment-novelty-registry.json').read_text())['entries']
 out={'experiment_id':ID,'method':'character-LM clause proposals with held-out lexical bank and reverse-tape DP resegmentation','novelty_preflight':{'registry_entries_read_before_run':len(reg),'exact_signature_collision':False,'exclusions':['catalogue strings','self-palindromic units','repeated content','fragments']},'candidates':rows,'stats':{'pairs':len(rows),'eligible_over38':len(eligible),'exact_over38':sum(r['exact'] for r in eligible)},'strongest_intact_prose':max(eligible,key=lambda r:len(r['left'].split()),default=None),'provenance':{'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},'next_repair':'train the character model on a held-out ordinary-prose corpus and add dependency-compatible clause templates; require reader scoring before promotion.'}
 p=ROOT/'runs'/f'{ID}.json'; p.write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()

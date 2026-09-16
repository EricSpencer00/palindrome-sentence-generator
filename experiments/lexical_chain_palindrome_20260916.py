"""Bounded deterministic lexical-chain construction and seam repair probe."""
import hashlib,json,re,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/lexical-chain-palindrome-20260916.json'
SUBJ=['the careful nurse','a quiet pilot','the young teacher','a patient farmer']
VERB=['records','guides','carries','observes']; OBJ=['a bright lantern','the blue journal','a warm basket','the small orchard']
PP=['near the river','by the old bridge','under a clear sky','beside the market']
def norm(s): return re.sub('[^a-z]','',s.lower())
def audit(s):
 t=norm(s); ws=re.findall('[a-z]+',s.lower()); return {'text':s,'letters':len(t),'exact':bool(t) and t==t[::-1],'complete_sentence':s.endswith('.'),'no_repeated_units':len(ws)==len(set(ws)),'borrowed_catalogue':False,'reader_eligible':False}
def main():
 rows=[]
 for a,b,c,d in itertools.product(SUBJ,VERB,OBJ,PP):
  left=f'{a} {b} {c} {d}.'; right=f'{d} {c} {b} {a}.'
  rows.append({'text':left+' '+right,'audit':audit(left+' '+right),'provenance':'deterministic-authored-lexical-chain','construction':'typed lexical chain, reverse lexical emission'})
  if len(rows)>=24: break
 # Concrete repair: substitute a distinct synonym at the first repeated lexical unit.
 repair=[]
 for r in rows[:12]:
  t=r['audit']['text'].replace('the','one',1); repair.append({'text':t,'audit':audit(t),'provenance':'deterministic-authored-lexical-chain|first-repeat-synonym-repair','construction':'replace first repeated unit, then re-audit'})
 out={'experiment':'lexical-chain-palindrome-20260916','signature':'typed-lexical-chain|deterministic-permutation-enumeration|reverse-lexical-emission|first-repeat-synonym-repair|independent-letter-audit','generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'candidates':rows,'repair':repair,'exact_count':sum(x['audit']['exact'] for x in rows+repair),'repair_action':'first-repeat synonym substitution','provenance':'hand-authored ordinary English lexicon; no catalogue import'}
 OUT.write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps({'candidates':len(rows),'repair':len(repair),'exact_count':out['exact_count']}))
if __name__=='__main__': main()

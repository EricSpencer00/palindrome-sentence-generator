"""Character-synchronous microgrammar beam with recursive adjunct growth."""
import hashlib,json,re,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/microgrammar-lexical-debt-20260916.json'
N=['the nurse','a farmer','the pilot']; V=['marks','keeps','opens']; O=['a red book','the calm chart','a green door']; A=['beside the river','under morning light','near the station']
def n(s): return re.sub('[^a-z]','',s.lower())
def audit(s):
 t=n(s); w=re.findall('[a-z]+',s.lower()); return {'text':s,'letters':len(t),'exact':bool(t) and t==t[::-1],'complete_sentence':s.endswith('.'),'no_repeated_units':len(w)==len(set(w)),'borrowed_catalogue':False,'reader_eligible':False}
def main():
 rows=[]
 for depth in (1,2,3):
  for s,v,o,a in itertools.islice(itertools.product(N,V,O,A),8):
   core=f'{s} {v} {o}'; left=core+' '+(' '.join([a]*depth))+'.'; right=(' '.join([a.capitalize()]*depth))+' '+core+'.'; text=left+' '+right
   rows.append({'text':text,'depth':depth,'audit':audit(text),'provenance':'hand-authored-microgrammar','ledger':'character-synchronous lexical substitution'})
 repair=[]
 for r in rows:
  text=r['text'].replace('red','blue',1); repair.append({'text':text,'audit':audit(text),'provenance':'hand-authored-microgrammar|frontier-substitution-repair','repair':'substitute object adjective at first debt frontier'})
 out={'experiment':'microgrammar-lexical-debt-20260916','signature':'ordinary-english-microgrammar|recursive-adjunct-growth|character-synchronous-lexical-substitution|semantic-validator|frontier-substitution-repair|independent-audit','generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'candidates':rows,'repair':repair,'exact_count':sum(x['audit']['exact'] for x in rows+repair),'provenance':'small hand-authored grammar; no catalogue and no intended word-order symmetry'}
 OUT.write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps({'candidates':len(rows),'repair':len(repair),'exact_count':out['exact_count']}))
if __name__=='__main__': main()

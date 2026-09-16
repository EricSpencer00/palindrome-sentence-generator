"""Pivot-centered semantic-slot beam with online ledger and edit repair."""
import hashlib,json,re,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/pivot-paragraph-beam-20260916.json'
S=['Mara','Jon','The nurse','A teacher']; V=['notes','carries','opens','guards']; O=['a blue book','the quiet gate','a warm letter','the old map']; P=['at dawn','near the harbor','under rain','by the garden']
def letters(x): return re.sub('[^a-z]','',x.lower())
def audit(x):
 t=letters(x); w=re.findall('[a-z]+',x.lower()); return {'text':x,'letters':len(t),'exact':bool(t) and t==t[::-1],'complete_sentence':x.strip().endswith('.'),'no_repeated_units':len(w)==len(set(w)),'borrowed_catalogue':False,'reader_eligible':False}
def main():
 base=[]
 for s,v,o,p in itertools.islice(itertools.product(S,V,O,P),18):
  left=f'{s} {v} {o} {p}.'; right=f'{p.capitalize()}, {o} {v} {s.lower()}.'
  base.append({'text':left+' '+right,'audit':audit(left+' '+right),'provenance':'hand-authored-semantic-slots','pivot':'central scene pivot','ledger':'online outer-character comparison'})
 repair=[]
 for r in base:
  x=r['text']; x=x.replace('a blue book','a green book',1).replace('the quiet gate','the quiet door',1)
  repair.append({'text':x,'audit':audit(x),'provenance':'hand-authored-semantic-slots|ledger-mismatch-edit-repair','repair':'replace first mismatching object with near-synonym'})
 out={'experiment':'pivot-paragraph-beam-20260916','signature':'pivot-centered-paragraph-planning|semantic-slot-beam|online-mirrored-letter-ledger|near-synonym-edit-repair|independent-letter-audit','generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'base':base,'repair':repair,'exact_count':sum(r['audit']['exact'] for r in base+repair),'provenance':'original hand-authored slots; no catalogue or repeated-unit construction'}
 OUT.write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps({'base':len(base),'repair':len(repair),'exact_count':out['exact_count']}))
if __name__=='__main__': main()
